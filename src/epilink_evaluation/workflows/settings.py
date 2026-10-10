"""Declarative operating definitions; the registry is shared by sweeps and replay."""

from ..provenance import fingerprint
from ..scorers import SCORERS
from ..selection.search import initial_cutoffs, initial_resolutions

TREE_CUTOFF_FIELDS = {
    "raw": "genetic_threshold_snps",
    "dated": "temporal_threshold_days",
}


def leiden_definition(config, name, resolution):
    spec = SCORERS[name].spec
    settings = config["clustering"]["leiden"]
    return {
        "score_name": name,
        "data_process": spec.data_process,
        "threshold": None,
        "empty": False,
        "graph_mode": "full",
        "kind": "leiden",
        "objective": settings["objective"],
        "resolution": float(resolution),
        "restarts": settings["restarts"],
        "algorithm_seed": settings["seed"],
        "pipeline": f"leiden/{name}",
    }


def treecluster_definition(config, process, kind, method, value):
    """Keep the searched cutoff in input units alongside the executed tree units."""
    alignment_length = config["simulation"].get(
        "alignment_length", config["simulation"]["sequence_length"]
    )
    value = int(value) if kind == "raw" else float(value)
    return {
        "kind": "treecluster",
        "tree_kind": kind,
        "data_process": process,
        "method": method,
        "threshold_input": value,
        "threshold": value / alignment_length if kind == "raw" else value,
        "threshold_units": "snps" if kind == "raw" else "days",
        "pipeline": f"treecluster/{process}/{kind}",
    }


def settings_registry(config, pairwise_thresholds=None):
    """Pairwise cutoffs come from development scores; clustering grids are independent."""
    definitions = {}

    def add(definition):
        key = fingerprint(definition)[:20]
        definitions[key] = definition

    for name in config["scorers"]:
        spec = SCORERS[name].spec
        if pairwise_thresholds is not None:
            pair_thresholds = [None, *sorted(set(pairwise_thresholds[name]))]
        else:
            pair_thresholds = []  # Not known until all development curves exist.
        for threshold in pair_thresholds:
            add(
                {
                    "score_name": name,
                    "data_process": spec.data_process,
                    "threshold": threshold,
                    "empty": threshold is None,
                    "kind": "pairwise",
                    "pipeline": f"pairwise/{name}",
                }
            )
        for threshold in pair_thresholds:
            base = {
                "score_name": name,
                "data_process": spec.data_process,
                "threshold": threshold,
                "empty": threshold is None,
            }
            if "components" in config["clustering"]["algorithms"]:
                add(
                    {
                        **base,
                        "kind": "components",
                        "pipeline": f"components/{name}",
                    }
                )
        if "leiden" in config["clustering"]["algorithms"]:
            leiden = config["clustering"]["leiden"]
            for resolution in initial_resolutions(leiden["resolutions"]):
                add(leiden_definition(config, name, resolution))
    tc = config["treecluster"]
    if tc["enabled"]:
        processes = sorted(
            {SCORERS[name].spec.data_process for name in config["scorers"]}
        )
        for process in processes:
            for kind, field in TREE_CUTOFF_FIELDS.items():
                for method in tc["methods"]:
                    for value in initial_cutoffs(tc[field], integer=kind == "raw"):
                        add(
                            treecluster_definition(config, process, kind, method, value)
                        )
    return definitions
