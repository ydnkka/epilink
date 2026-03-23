from __future__ import annotations

import os
import sys
from unittest.mock import patch

import pandas as pd

from epilink.cli import main

# Create a sample CSV
df = pd.DataFrame({"sample_time_difference": [1.0, 2.0, 3.0], "genetic_distance": [0, 1, 2]})
df.to_csv("test_input.csv", index=False)

test_args = ["epilink", "test_input.csv", "--output", "test_output.csv", "--mc-samples", "100"]
with patch.object(sys, "argv", test_args):
    main()

# Check output
if os.path.exists("test_output.csv"):
    output_df = pd.read_csv("test_output.csv")
    print("CLI executed successfully. Output columns:")
    print(output_df.columns)
    print(output_df)
else:
    print("CLI failed to produce output.")

# Clean up
os.remove("test_input.csv")
os.remove("test_output.csv")
