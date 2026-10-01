"""Generation parameter conversion, independent of study configuration loaders."""


def natural_history(params):
    from epilink import NaturalHistoryParameters

    expanded = {
        k: v for k, v in params.items() if k not in ("incubation", "testing_delay")
    }
    for name in ("incubation", "testing_delay"):
        shape = 1 / params[name]["cv"] ** 2
        expanded[f"{name}_shape"] = shape
        expanded[f"{name}_scale"] = params[name]["mean"] / shape
    return NaturalHistoryParameters(**expanded)
