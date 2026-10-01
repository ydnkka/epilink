"""Declarative operating definitions; the registry is shared by sweeps and replay."""

from ..provenance import fingerprint
from ..scorers import SCORERS


def settings_registry(config):
    definitions = {}

    def add(definition):
        key = fingerprint(definition)[:20]
        definitions[key] = definition

    for name in config["scorers"]:
        spec = SCORERS[name].spec
        thresholds = [None, *sorted(set(config["thresholds"][spec.family]))]
        for threshold in thresholds:
            base = {
                "score_name": name,
                "data_process": spec.data_process,
                "threshold": threshold,
                "empty": threshold is None,
            }
            add({**base, "kind": "pairwise", "pipeline": f"pairwise/{name}"})
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
                    for resolution in leiden["resolutions"]:
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
        sequence_length = config["simulation"]["sequence_length"]
        for process in processes:
            for kind, thresholds, units in (
                ("raw", tc["genetic_thresholds"], "substitutions_per_site"),
                ("dated", tc["threshold_days"], "days"),
            ):
                for method in tc["methods"]:
                    for threshold in thresholds:
                        threshold_value = (
                            float(threshold) / sequence_length
                            if kind == "raw"
                            else float(threshold)
                        )
                        add(
                            {
                                "kind": "treecluster",
                                "tree_kind": kind,
                                "data_process": process,
                                "method": method,
                                "threshold": threshold_value,
                                "threshold_units": units,
                                "days_per_year": tc["days_per_year"],
                                "pipeline": f"treecluster/{process}/{kind}",
                            }
                        )
    return definitions
