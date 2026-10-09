from __future__ import annotations

import copy
import io
import math
import os
import re
import shutil
import subprocess
import tempfile
import textwrap
from collections.abc import Hashable, Mapping
from datetime import date, datetime
from numbers import Integral, Real
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import networkx as nx
import numpy as np
import pandas as pd

from ..model import ConfigurationError, EpiLinkError
from .genome import _PACK_SHIFTS, PackedGenomicData
from .results import PhylogenyResult, SimulationResult

if TYPE_CHECKING:
    from Bio.Phylo.BaseTree import Tree

_REFERENCE_ID = "epilink_reference"
_DATE_COLUMNS = ["node", "case_id", "is_tip", "date", "sample_date"]
_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_DATE_PATTERN = re.compile(rf'(?:^|[&,])\s*date\s*=\s*"?({_NUMBER})"?(?=,|$)')
_RATE_PATTERN = re.compile(rf"^\s*rate\s+({_NUMBER})(?=\s|,|$)", re.MULTILINE)
_DNA_ALPHABET = frozenset("ACGTRYSWKMBDHVN-?")
_DateSource = Mapping[str, float | str | date] | str | os.PathLike[str]


class PhylogenyError(EpiLinkError, RuntimeError):
    """Raised when phylogenetic inference or dating fails."""


def _find_iqtree(executable: str | os.PathLike[str] | None) -> str:
    candidates = (
        [os.fspath(executable)] if executable is not None else ["iqtree3", "iqtree2", "iqtree"]
    )
    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved is not None:
            return str(Path(resolved).resolve())
    raise PhylogenyError(
        "IQ-TREE was not found. Install IQ-TREE >=2.0.6 and add it to PATH, "
        "or supply iqtree_executable with the executable path."
    )


def _decode_sequence(packed: PackedGenomicData, index: int) -> str:
    blocks = packed.packed_u64[index]
    codes = ((blocks[:, np.newaxis] >> _PACK_SHIFTS) & np.uint64(3)).reshape(-1)
    bases = np.array([packed.bases_map[i] for i in range(4)], dtype="<U1")
    return "".join(bases[codes[: packed.original_length].astype(np.intp)])


def _prepare_samples(
    simulation: SimulationResult,
    epidemic_tree: nx.DiGraph,
    sequence_model: str,
    dated: bool,
    clock_rate: float | None,
) -> tuple[list[Hashable], list[str], dict[str, float], str]:
    if sequence_model not in {"deterministic", "stochastic"}:
        raise ConfigurationError("sequence_model must be 'deterministic' or 'stochastic'.")
    packed = simulation.packed[sequence_model]
    reference = simulation.reference_sequence
    if (
        packed.original_length <= 0
        or reference.shape != (packed.original_length,)
        or not np.issubdtype(reference.dtype, np.integer)
        or np.any((reference < 0) | (reference > 3))
        or packed.packed_u64.shape[1] * 32 < packed.original_length
        or packed.bases_map != {0: "A", 1: "C", 2: "G", 3: "T"}
    ):
        raise ConfigurationError(
            "Packed sequences and reference must have matching A/C/G/T genomes."
        )

    nodes = [node for node, data in epidemic_tree.nodes(data=True) if data.get("sampled", False)]
    if len(nodes) < 3:
        raise ConfigurationError("At least three sampled cases are required, plus the reference.")
    labels = [str(node) for node in nodes]
    if len(set(labels)) != len(labels) or any(
        not label or "\n" in label or "\r" in label for label in labels
    ):
        raise ConfigurationError(
            "Sampled case IDs must have distinct, nonempty, single-line labels."
        )

    sequences: list[str] = []
    dates: dict[str, float] = {}
    indices: set[int] = set()
    for node, label in zip(nodes, labels, strict=True):
        index = packed.node_to_idx.get(node)
        if not isinstance(index, Integral) or not 0 <= index < packed.n_seqs or index in indices:
            raise ConfigurationError(
                f"Missing or invalid sequence mapping for sampled case {node!r}."
            )
        indices.add(index)
        sequences.append(_decode_sequence(packed, index))
        if dated:
            date = epidemic_tree.nodes[node].get("sample_date")
            if isinstance(date, bool) or not isinstance(date, Real) or not math.isfinite(date):
                raise ConfigurationError(
                    f"Sampled case {node!r} needs a finite numeric sample_date."
                )
            dates[label] = float(date)
    if dated and clock_rate is None and len(set(dates.values())) < 2:
        raise ConfigurationError(
            "Distinct sampling dates or a fixed clock_rate are required for dating."
        )

    reference_name = "reference"
    suffix = 1
    while reference_name in labels:
        reference_name = f"reference_{suffix}"
        suffix += 1
    return nodes, sequences, dates, reference_name


def _restore_tips(tree: Tree, labels: dict[str, str], expected: set[str]) -> None:
    tips = tree.get_terminals()
    names = [tip.name for tip in tips]
    if set(names) != expected or len(names) != len(expected):
        raise PhylogenyError("IQ-TREE output does not contain the expected sampled taxa.")
    for clade in tree.find_clades():
        length = clade.branch_length
        if length is not None and (not math.isfinite(length) or length < 0):
            raise PhylogenyError("IQ-TREE/LSD2 output contains invalid branch lengths.")
    for tip in tips:
        tip.name = labels[tip.name]
    tree.rooted = True


def _node_dates(
    tree: Tree,
    nodes: list[Hashable],
    sample_dates: dict[str, float],
    date_origin: date | None = None,
) -> pd.DataFrame:
    case_ids = {str(node): node for node in nodes}
    used_names = set(case_ids)
    rows = []
    internal_index = 0
    for clade in tree.find_clades(order="preorder"):
        # Bio.Phylo's NEXUS reader retains the surrounding comment brackets.
        comment = (clade.comment or "").strip("[]")
        match = _DATE_PATTERN.search(comment)
        if match is None or not math.isfinite(float(match.group(1))):
            raise PhylogenyError("LSD2 output is missing finite node dates.")
        date = float(match.group(1))
        clade.comment = comment
        if not clade.is_terminal():
            name = f"internal_{internal_index}"
            while name in used_names:
                internal_index += 1
                name = f"internal_{internal_index}"
            clade.name = name
            used_names.add(name)
            internal_index += 1
        rows.append(
            {
                "node": clade.name,
                "case_id": case_ids.get(clade.name),
                "is_tip": clade.is_terminal(),
                "date": date,
                "sample_date": sample_dates.get(clade.name, np.nan),
            }
        )
    frame = pd.DataFrame(rows, columns=_DATE_COLUMNS)
    frame["case_id"] = pd.Series([row["case_id"] for row in rows], dtype=object)
    if date_origin is not None:
        origin = pd.Timestamp(date_origin)
        frame["calendar_date"] = origin + pd.to_timedelta(frame["date"], unit="D")
        sample_offsets = pd.to_timedelta(frame["sample_date"].fillna(0.0), unit="D")
        frame["sample_calendar_date"] = (origin + sample_offsets).where(
            frame["sample_date"].notna()
        )
        for clade, calendar_date in zip(
            tree.find_clades(order="preorder"), frame["calendar_date"], strict=True
        ):
            clade.comment += f',calendar_date="{calendar_date.isoformat()}"'
    return frame


def _validate_options(
    dated: bool,
    model: str,
    threads: int,
    seed: int,
    clock_rate: float | None,
    timeout: float | None,
) -> None:
    for name, value in (("threads", threads), ("seed", seed)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ConfigurationError(f"{name} must be a positive integer.")
    if not isinstance(model, str) or not model or any(char.isspace() for char in model):
        raise ConfigurationError("model must be a nonempty IQ-TREE model without whitespace.")
    for name, positive_number in (("clock_rate", clock_rate), ("timeout", timeout)):
        if positive_number is not None and (
            isinstance(positive_number, bool)
            or not isinstance(positive_number, Real)
            or not math.isfinite(positive_number)
            or positive_number <= 0
        ):
            raise ConfigurationError(f"{name} must be finite and positive.")
    if not dated and clock_rate is not None:
        raise ConfigurationError("clock_rate requires dated=True.")


def _read_fasta(filepath: str | os.PathLike[str]) -> dict[str, str]:
    try:
        from Bio import SeqIO
    except ImportError as error:
        raise ImportError(
            "Phylogeny inference requires Biopython: pip install 'epilink[phylogeny]'."
        ) from error
    records: dict[str, str] = {}
    path = Path(filepath).expanduser()
    try:
        with path.open(encoding="utf-8") as handle:
            for record in SeqIO.parse(handle, "fasta"):
                identifier = record.id
                if not identifier or identifier in records:
                    raise ConfigurationError(
                        f"FASTA IDs must be nonempty and unique: {identifier!r}."
                    )
                sequence = str(record.seq).upper()
                if not sequence or set(sequence) - _DNA_ALPHABET:
                    raise ConfigurationError(
                        f"Sequence {identifier!r} must contain aligned DNA (IUPAC bases, '-' or '?')."
                    )
                records[identifier] = sequence
    except (OSError, ValueError) as error:
        raise ConfigurationError(f"Cannot read aligned FASTA {path}: {error}") from error
    if not records or len({len(sequence) for sequence in records.values()}) != 1:
        raise ConfigurationError("FASTA must contain nonempty sequences of equal aligned length.")
    return records


def _load_dates(source: _DateSource) -> dict[str, object]:
    values: dict[str, object]
    if isinstance(source, Mapping):
        values = dict(source)
    else:
        path = Path(source).expanduser()
        try:
            table = pd.read_csv(
                path,
                sep="\t" if path.suffix.lower() in {".tsv", ".txt"} else ",",
                dtype=str,
                keep_default_na=False,
            )
        except (OSError, ValueError, pd.errors.ParserError) as error:
            raise ConfigurationError(f"Cannot read sampling dates {path}: {error}") from error
        if not {"case_id", "sample_date"} <= set(table.columns):
            raise ConfigurationError("Dates CSV/TSV needs case_id and sample_date columns.")
        if table["case_id"].duplicated().any():
            raise ConfigurationError("Dates file contains duplicate case_id values.")
        values = dict(zip(table["case_id"], table["sample_date"], strict=True))
    if any(not isinstance(identifier, str) or not identifier for identifier in values):
        raise ConfigurationError("Date identifiers must be nonempty strings matching FASTA IDs.")
    return values


def _parse_sample_date(value: object, identifier: str) -> float | date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, str):
        value = value.strip()
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            try:
                return date.fromisoformat(value)
            except ValueError as error:
                raise ConfigurationError(
                    f"Invalid calendar date for {identifier!r}: {value!r}."
                ) from error
        try:
            value = float(value)
        except ValueError as error:
            raise ConfigurationError(
                f"Invalid sampling date for {identifier!r}: {value!r}."
            ) from error
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ConfigurationError(f"{identifier!r} needs a finite numeric day or YYYY-MM-DD date.")
    return float(value)


def _normalise_dates(
    source: _DateSource | None,
    identifiers: list[Hashable],
    reference_name: str,
    clock_rate: float | None,
) -> tuple[dict[str, float], date | None]:
    if source is None:
        raise ConfigurationError("dates is required when dated=True.")
    values = _load_dates(source)
    names = {str(identifier) for identifier in identifiers}
    missing = names - values.keys()
    extra = values.keys() - names - {reference_name}
    if missing or extra:
        raise ConfigurationError(
            f"Date IDs do not match the alignment: missing={sorted(missing)}, extra={sorted(extra)}."
        )
    parsed = {
        identifier: _parse_sample_date(values[identifier], identifier) for identifier in names
    }
    calendar_dates = {
        identifier: value for identifier, value in parsed.items() if isinstance(value, date)
    }
    origin = None
    if calendar_dates:
        if len(calendar_dates) != len(parsed):
            raise ConfigurationError("Use either numeric days or calendar dates, not a mixture.")
        origin = min(calendar_dates.values())
        normalised = {
            identifier: float((value - origin).days) for identifier, value in calendar_dates.items()
        }
    else:
        normalised = {
            identifier: value for identifier, value in parsed.items() if isinstance(value, float)
        }
    if clock_rate is None and len(set(normalised.values())) < 2:
        raise ConfigurationError(
            "Distinct sampling dates or a fixed clock_rate are required for dating."
        )
    return normalised, origin


def build_phylogenetic_tree_from_fasta(
    alignment_fasta: str | os.PathLike[str],
    *,
    reference_id: str | None = None,
    reference_fasta: str | os.PathLike[str] | None = None,
    dates: _DateSource | None = None,
    dated: bool = True,
    output_dir: str | os.PathLike[str] = "phylogeny",
    model: str = "MFP",
    threads: int = 1,
    seed: int = 2026,
    clock_rate: float | None = None,
    iqtree_executable: str | os.PathLike[str] | None = None,
    timeout: float | None = None,
) -> PhylogenyResult:
    """Infer genetic and dated trees from an aligned DNA FASTA.

    Supply exactly one of ``reference_id`` (an ID inside the alignment) or
    ``reference_fasta`` (a separate aligned, single-record FASTA). The reference
    must have the same alignment columns as the sample sequences. IDs are the
    first whitespace-delimited token of FASTA headers. At least three sample
    sequences, excluding the reference, are required. Gaps and IUPAC ambiguity
    bases are preserved; no alignment or gap filtering is performed.

    ``dates`` is an ID-to-date mapping or CSV/TSV path with ``case_id`` and
    ``sample_date`` columns. Every sample needs a date when ``dated=True``.
    Values must be finite numeric days, or all YYYY-MM-DD strings/``date``
    objects. Calendar dates are normalised to days from the earliest sample;
    the result retains ``date_origin`` and calendar node/sample dates. Numeric
    origins are preserved. An optional reference date is ignored: the reference
    is used for rooting and excluded from the dated tree. With ``dated=False``,
    dates are not needed or read.

    ``model="MFP"`` selects a substitution model with IQ-TREE ModelFinder.
    Other options and requirements match :func:`build_phylogenetic_tree`.
    In particular, fixed/returned clock rates are substitutions/site/day for
    either date format, and dated branch lengths are always days. Separate
    run directories preserve source FASTA/date files and prior inference runs.
    """
    _validate_options(dated, model, threads, seed, clock_rate, timeout)
    if (reference_id is None) == (reference_fasta is None):
        raise ConfigurationError("Supply exactly one of reference_id or reference_fasta.")
    records = _read_fasta(alignment_fasta)
    if reference_fasta is not None:
        reference_records = _read_fasta(reference_fasta)
        if len(reference_records) != 1:
            raise ConfigurationError("reference_fasta must contain exactly one sequence.")
        reference_name, reference_sequence = next(iter(reference_records.items()))
        if reference_name in records:
            raise ConfigurationError(
                "Reference ID already exists in the alignment; use reference_id."
            )
        if len(reference_sequence) != len(next(iter(records.values()))):
            raise ConfigurationError(
                "Reference and sample sequences must have equal aligned length."
            )
    else:
        if reference_id not in records:
            raise ConfigurationError(f"Reference ID {reference_id!r} is not in the alignment.")
        reference_name = str(reference_id)
        reference_sequence = records.pop(reference_name)
    nodes: list[Hashable] = list(records)
    if len(nodes) < 3:
        raise ConfigurationError(
            "At least three sample sequences are required, plus the reference."
        )
    normalised_dates, origin = (
        _normalise_dates(dates, nodes, reference_name, clock_rate) if dated else ({}, None)
    )
    return _run_phylogeny(
        nodes,
        list(records.values()),
        reference_sequence,
        reference_name,
        normalised_dates,
        dated=dated,
        output_dir=output_dir,
        model=model,
        threads=threads,
        seed=seed,
        clock_rate=clock_rate,
        iqtree_executable=iqtree_executable,
        timeout=timeout,
        date_origin=origin,
    )


def build_phylogenetic_tree(
    simulation: SimulationResult,
    epidemic_tree: nx.DiGraph,
    *,
    sequence_model: Literal["deterministic", "stochastic"] = "stochastic",
    dated: bool = True,
    output_dir: str | os.PathLike[str] = "phylogeny",
    model: str = "JC",
    threads: int = 1,
    seed: int = 2026,
    clock_rate: float | None = None,
    iqtree_executable: str | os.PathLike[str] | None = None,
    timeout: float | None = None,
) -> PhylogenyResult:
    """Infer reference-rooted sampled phylogenies using IQ-TREE and LSD2.

    Parameters
    ----------
    simulation : SimulationResult
        Simulated genomes. Packed data are decoded; raw arrays are not required.
    epidemic_tree : networkx.DiGraph
        Source of ``sampled`` flags and numeric ``sample_date`` values in days.
        Transmission edges are not used for phylogenetic inference.
    sequence_model : {"deterministic", "stochastic"}, default="stochastic"
        Sequence set to infer from. At least three sampled cases are required.
    dated : bool, default=True
        Also construct a time-scaled phylogeny using sampling dates.
    output_dir : str or path-like, default="phylogeny"
        Parent of a unique run directory containing alignments, trees and logs.
    model : str, default="JC"
        IQ-TREE substitution model. JC matches the simulation's symmetric base
        changes; use e.g. ``"GTR+G"`` or ``"MFP"`` for other models/model selection.
    threads : int, default=1
        Number of IQ-TREE threads.
    seed : int, default=2026
        Positive random seed for reproducible inference.
    clock_rate : float, optional
        Fixed dating rate in substitutions/site/day. By default LSD2 estimates
        the rate from distinct tip dates. Same-day tips require a fixed rate.
    iqtree_executable : str or path-like, optional
        Executable name/path; otherwise discover iqtree3, iqtree2 or iqtree.
        IQ-TREE >=2.0.6 is required and must be installed separately.
    timeout : float, optional
        Maximum runtime in seconds; no limit by default.

    Returns
    -------
    PhylogenyResult
        Biopython genetic and optional dated trees, node dates, clock rate and
        artifact paths. Genetic lengths are substitutions/site, dated lengths
        are days. The genetic tree includes the reference; LSD2 removes the
        undated reference from the dated tree after rooting. Dates retain the
        simulation time origin. Original case labels are restored in exported
        trees; ``taxon_labels.tsv`` maps safe alignment IDs to case IDs.

    Raises
    ------
    ConfigurationError
        Invalid options, sequences, sample mappings or dates.
    ImportError
        Biopython is unavailable; install ``epilink[phylogeny]``.
    PhylogenyError
        IQ-TREE is unavailable, fails, times out, or produces invalid outputs.
    """
    _validate_options(dated, model, threads, seed, clock_rate, timeout)
    nodes, sequences, dates, reference_name = _prepare_samples(
        simulation, epidemic_tree, sequence_model, dated, clock_rate
    )
    return _run_phylogeny(
        nodes,
        sequences,
        simulation.reference_sequence_string,
        reference_name,
        dates,
        dated=dated,
        output_dir=output_dir,
        model=model,
        threads=threads,
        seed=seed,
        clock_rate=clock_rate,
        iqtree_executable=iqtree_executable,
        timeout=timeout,
    )


def _run_phylogeny(
    nodes: list[Hashable],
    sequences: list[str],
    reference_sequence: str,
    reference_name: str,
    dates: dict[str, float],
    *,
    dated: bool,
    output_dir: str | os.PathLike[str],
    model: str,
    threads: int,
    seed: int,
    clock_rate: float | None,
    iqtree_executable: str | os.PathLike[str] | None,
    timeout: float | None,
    date_origin: date | None = None,
) -> PhylogenyResult:
    try:
        from Bio import Phylo
        from Bio.Nexus.Nexus import NexusError
        from Bio.Phylo.NewickIO import NewickError
    except ImportError as error:
        raise ImportError(
            "Phylogeny inference requires Biopython: pip install 'epilink[phylogeny]'."
        ) from error
    executable = _find_iqtree(iqtree_executable)

    parent = Path(output_dir).expanduser().resolve()
    parent.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="run-", dir=parent))
    paths = {
        "alignment": run_dir / "sampled.fasta",
        "taxon_labels": run_dir / "taxon_labels.tsv",
        "log": run_dir / "inference.log",
        "iqtree_report": run_dir / "iqtree.iqtree",
        "iqtree_tree": run_dir / "iqtree.treefile",
        "raw_tree": run_dir / "raw_tree.nwk",
    }
    labels = {f"epilink_tip_{i}": str(node) for i, node in enumerate(nodes)}
    labels[_REFERENCE_ID] = reference_name
    with paths["alignment"].open("w", encoding="utf-8") as fasta:
        for alias, sequence in zip(labels, [*sequences, reference_sequence], strict=True):
            fasta.write(f">{alias}\n{textwrap.fill(sequence, width=100)}\n")
    pd.DataFrame({"taxon": list(labels), "case_id": list(labels.values())}).to_csv(
        paths["taxon_labels"], sep="\t", index=False
    )
    # Relative paths also work around LSD2's command parser for directories with spaces.
    command = [
        executable,
        "-s",
        "sampled.fasta",
        "-st",
        "DNA",
        "-m",
        model,
        "-o",
        _REFERENCE_ID,
        "-T",
        str(threads),
        "-seed",
        str(seed),
        "-keep-ident",
        "--prefix",
        "iqtree",
    ]
    if dated:
        paths.update(
            {
                "sampling_dates": run_dir / "sampling_dates.txt",
                "lsd_report": run_dir / "iqtree.timetree.lsd",
                "lsd_tree": run_dir / "iqtree.timetree.nex",
                "dated_tree": run_dir / "dated_tree.nwk",
                "dated_nexus": run_dir / "dated_tree.nex",
                "node_dates": run_dir / "node_dates.tsv",
            }
        )
        if date_origin is not None:
            paths["date_origin"] = run_dir / "date_origin.txt"
            paths["date_origin"].write_text(f"{date_origin.isoformat()}\n", encoding="utf-8")
        with paths["sampling_dates"].open("w", encoding="utf-8") as date_file:
            for alias, label in labels.items():
                if alias != _REFERENCE_ID:
                    date_file.write(f"{alias}\t{dates[label]:.17g}\n")
        options = "-G -D 1 -R 1 -l 0 -u 0 -U 0"
        if clock_rate is not None:
            paths["clock_rate"] = run_dir / "clock_rate.txt"
            paths["clock_rate"].write_text(f"{clock_rate:.17g}\n", encoding="utf-8")
            options += " -w clock_rate.txt"
        command += ["--date", "sampling_dates.txt", "--date-options", options]

    with paths["log"].open("w", encoding="utf-8") as log:
        try:
            completed = subprocess.run(
                command,
                cwd=run_dir,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise PhylogenyError(
                f"IQ-TREE could not complete: {error}. See {paths['log']}."
            ) from error
    if completed.returncode != 0:
        raise PhylogenyError(
            f"IQ-TREE exited with status {completed.returncode}. See {paths['log']}."
        )

    try:
        raw_tree = Phylo.read(paths["iqtree_tree"], "newick")
        _restore_tips(raw_tree, labels, set(labels))
        raw_tree.root_with_outgroup(reference_name)
        Phylo.write(raw_tree, paths["raw_tree"], "newick", format_branch_length="%1.12g")
        dated_tree = None
        node_dates = pd.DataFrame(columns=_DATE_COLUMNS)
        inferred_rate = None
        if dated:
            # LSD2's .timetree.nwk is in substitution units; its NEXUS has time lengths.
            dated_tree = Phylo.read(paths["lsd_tree"], "nexus")
            _restore_tips(dated_tree, labels, set(labels) - {_REFERENCE_ID})
            node_dates = _node_dates(dated_tree, nodes, dates, date_origin)
            rate_match = _RATE_PATTERN.search(paths["lsd_report"].read_text(encoding="utf-8"))
            if rate_match is None:
                raise PhylogenyError("LSD2 did not report a dating rate.")
            inferred_rate = float(rate_match.group(1))
            if not math.isfinite(inferred_rate) or inferred_rate <= 0:
                raise PhylogenyError("LSD2 reported an invalid dating rate.")
            Phylo.write(dated_tree, paths["dated_tree"], "newick", format_branch_length="%1.12g")
            # A NEXUS translate block preserves quoted case labels across readers.
            nexus_tree = copy.deepcopy(dated_tree)
            translations = []
            for i, tip in enumerate(nexus_tree.get_terminals(), start=1):
                label = tip.name.replace("'", "''")
                translations.append(f"{i} '{label}'")
                tip.name = str(i)
            newick = io.StringIO()
            Phylo.write(nexus_tree, newick, "newick", format_branch_length="%1.12g")
            translation = ",\n".join(translations)
            paths["dated_nexus"].write_text(
                f"#NEXUS\nBegin trees;\nTranslate\n{translation};\n"
                f"Tree dated = [&R] {newick.getvalue()}End;\n",
                encoding="utf-8",
            )
            node_dates.to_csv(paths["node_dates"], sep="\t", index=False)
    except (OSError, ValueError, OverflowError, NewickError, NexusError, PhylogenyError) as error:
        raise PhylogenyError(
            f"Invalid IQ-TREE/LSD2 output: {error}. See {paths['log']}."
        ) from error

    return PhylogenyResult(
        raw_tree=raw_tree,
        dated_tree=dated_tree,
        node_dates=node_dates,
        output_paths=paths,
        sample_ids=tuple(nodes),
        reference_name=reference_name,
        clock_rate=inferred_rate,
        date_origin=date_origin,
    )


__all__ = [
    "PhylogenyError",
    "PhylogenyResult",
    "build_phylogenetic_tree",
    "build_phylogenetic_tree_from_fasta",
]
