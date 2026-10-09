from __future__ import annotations

from collections.abc import Hashable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Generic, Literal, TypeVar, overload

import numpy as np
import numpy.typing as npt
import pandas as pd

from .genome import PackedGenomicData

if TYPE_CHECKING:
    from Bio.Phylo.BaseTree import Tree

T = TypeVar("T", covariant=True)
NDArrayInt8 = npt.NDArray[np.int8]


@dataclass(frozen=True, slots=True)
class SimulationSequenceSet(Generic[T]):
    """Paired deterministic and stochastic simulation outputs."""

    deterministic: T
    stochastic: T

    def __getitem__(self, key: str) -> T:
        if key not in {"deterministic", "stochastic"}:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(("deterministic", "stochastic"))

    def __len__(self) -> int:
        return 2

    def __contains__(self, key: object) -> bool:
        return key in {"deterministic", "stochastic"}

    def to_dict(self) -> dict[str, T]:
        return {"deterministic": self.deterministic, "stochastic": self.stochastic}


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Typed result returned by :func:`simulate_genomic_sequences`.

    ``reference_sequence`` is the original, unmutated one-dimensional integer
    sequence, encoded as 0=A, 1=C, 2=G, and 3=T. It is always included, even when
    raw node sequences are not requested. ``reference_sequence_string`` exposes
    the same reference decoded as an A/C/G/T string for FASTA export.
    """

    packed: SimulationSequenceSet[PackedGenomicData]
    raw: SimulationSequenceSet[NDArrayInt8] | None
    reference_sequence: NDArrayInt8

    @property
    def reference_sequence_string(self) -> str:
        """Return the original reference decoded as an A/C/G/T string."""
        base_lookup = np.array(["A", "C", "G", "T"], dtype="<U1")
        return "".join(base_lookup[self.reference_sequence])

    @overload
    def __getitem__(self, key: Literal["packed"]) -> SimulationSequenceSet[PackedGenomicData]: ...

    @overload
    def __getitem__(self, key: Literal["raw"]) -> SimulationSequenceSet[NDArrayInt8] | None: ...

    @overload
    def __getitem__(self, key: Literal["reference_sequence"]) -> NDArrayInt8: ...

    @overload
    def __getitem__(self, key: Literal["reference_sequence_string"]) -> str: ...

    @overload
    def __getitem__(
        self, key: str
    ) -> (
        SimulationSequenceSet[PackedGenomicData]
        | SimulationSequenceSet[NDArrayInt8]
        | NDArrayInt8
        | str
        | None
    ): ...

    def __getitem__(
        self, key: str
    ) -> (
        SimulationSequenceSet[PackedGenomicData]
        | SimulationSequenceSet[NDArrayInt8]
        | NDArrayInt8
        | str
        | None
    ):
        if key not in {"packed", "raw", "reference_sequence", "reference_sequence_string"}:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(("packed", "raw", "reference_sequence", "reference_sequence_string"))

    def __len__(self) -> int:
        return 4

    def __contains__(self, key: object) -> bool:
        return key in {"packed", "raw", "reference_sequence", "reference_sequence_string"}

    def to_dict(self) -> dict[str, object]:
        return {
            "packed": self.packed.to_dict(),
            "raw": None if self.raw is None else self.raw.to_dict(),
            "reference_sequence": self.reference_sequence,
            "reference_sequence_string": self.reference_sequence_string,
        }


@dataclass(frozen=True, slots=True)
class PhylogenyResult:
    """Sequence-based and optionally dated phylogenies of sampled cases.

    Trees are Biopython ``Tree`` objects. ``raw_tree`` has branch lengths in
    substitutions/site and includes the reference outgroup. ``dated_tree`` has
    branch lengths in days and excludes the undated reference. ``node_dates``
    contains ``node``, ``case_id``, ``is_tip``, ``date``, and ``sample_date``
    columns, with dates relative to the original simulation time origin.
    ``clock_rate`` is in substitutions/site/day, or ``None`` without dating.
    """

    raw_tree: Tree
    dated_tree: Tree | None
    node_dates: pd.DataFrame
    output_paths: dict[str, Path]
    sample_ids: tuple[Hashable, ...]
    reference_name: str
    clock_rate: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "raw_tree": self.raw_tree,
            "dated_tree": self.dated_tree,
            "node_dates": self.node_dates,
            "output_paths": self.output_paths,
            "sample_ids": self.sample_ids,
            "reference_name": self.reference_name,
            "clock_rate": self.clock_rate,
        }


__all__ = ["PhylogenyResult", "SimulationResult", "SimulationSequenceSet"]
