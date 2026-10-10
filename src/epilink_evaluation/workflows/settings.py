"""Declarative operating definitions; the registry is shared by sweeps and replay."""

from ..provenance import fingerprint
from ..scorers import SCORERS


def settings_registry(config, pairwise_thresholds=None):
    """Graph grids stay finite; exact pairwise candidates come from development."""
    definitions = {}

    def add(definition):
        key = fingerprint(definition)[:20]
        definitions[key] = definition

    for name in config["scorers"]:
        spec = SCORERS[name].spec
        thresholds = [None, *sorted(set(config["thresholds"][spec.family]))]
        if config["pairwise"].get("threshold_mode", "configured") == "configured":
            pair_thresholds = thresholds
        elif pairwise_thresholds is not None:
            pair_thresholds = [None, *sorted(set(pairwise_thresholds[name]))]
        else:
            pair_thresholds = []  # Not known until all development curves exist.
        for threshold in pair_thresholds:
            add({
                "score_name": name,
                "data_process": spec.data_process,
                "threshold": threshold,
                "empty": threshold is None,
                "kind": "pairwise",
                "pipeline": f"pairwise/{name}",
            })
        for threshold in thresholds:
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
                        "weight_policy": "binary",
                        "pipeline": f"components/{name}",
                    }
                )
            if "leiden" in config["clustering"]["algorithms"]:
                leiden = config["clustering"]["leiden"]
                for policy in leiden["weight_policies"]:
                    if policy == "native" and spec.family == "genetic":
                        continue
                    resolutions = leiden.get("resolutions_by_weight_policy", {}).get(
                        policy, leiden["resolutions"]
                    )
                    for resolution in sorted(set(resolutions)):
                        add(
                            {
                                **base,
                                "kind": "leiden",
                                "weight_policy": policy,
                                "objective": leiden["objective"],
                                "resolution": float(resolution),
                                "restarts": leiden["restarts"],
                                "algorithm_seed": leiden["seed"],
                                "pipeline": f"leiden/{name}/{policy}",
                            }
                        )
    tc = config["treecluster"]
    if tc["enabled"]:
        processes = sorted(
            {SCORERS[name].spec.data_process for name in config["scorers"]}
        )
        alignment_length = config["simulation"].get(
            "alignment_length", config["simulation"]["sequence_length"]
        )
        for process in processes:
            for kind, thresholds, units in (
                ("raw", tc["genetic_threshold_snps"], "snps"),
                ("dated", tc["threshold_days"], "days"),
            ):
                for method in tc["methods"]:
                    for snp_count in thresholds:
                        threshold_value = (
                            float(snp_count) / alignment_length
                            if kind == "raw"
                            else float(snp_count)
                        )
                        add(
                            {
                                "kind": "treecluster",
                                "tree_kind": kind,
                                "data_process": process,
                                "method": method,
                                "threshold": threshold_value,
                                "threshold_units": units,
                                "pipeline": f"treecluster/{process}/{kind}",
                            }
                        )
    return definitions
