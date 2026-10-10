from __future__ import annotations

import io

import pandas as pd

from ..phylogeny.external import command_identity, run_command, treecluster_executable
from ..phylogeny.trees import validate_tree
from ..schemas import validate_partition


def treecluster(tree_path, cases, method, threshold, config, directory):
    validate_tree(tree_path, cases.case_id)
    tool = command_identity(treecluster_executable(config))
    argv = [tool["path"], "-i", str(tree_path), "-t", str(threshold), "-m", method]
    output = run_command(
        argv, directory, "treecluster",
        config.get("timeout", config.get("command_timeout_seconds", 1800)),
    )
    frame = pd.read_csv(io.StringIO(output), sep="\t", dtype={"SequenceName": str})
    if list(frame.columns) != ["SequenceName", "ClusterNumber"]:
        raise ValueError("Unexpected TreeCluster output schema")
    assignments = [
        f"singleton:{case}" if int(cluster) == -1 else f"cluster:{int(cluster)}"
        for case, cluster in zip(frame.SequenceName, frame.ClusterNumber)
    ]
    memberships = pd.DataFrame(
        {"case_id": frame.SequenceName, "cluster_id": assignments}
    )
    labels = validate_partition(cases, memberships)
    return labels, {
        "command": argv,
        "executable": tool,
        "threshold_tree_units": threshold,
    }
