from __future__ import annotations

import csv
import io
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from epilink.cli import main

class TestCLI(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = TemporaryDirectory()
        self.input_path = Path(self.temp_dir.name) / "input.csv"
        self.output_path = Path(self.temp_dir.name) / "output.csv"

        with open(self.input_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["sample_time_difference", "genetic_distance"])
            writer.writerow([1.0, 0])
            writer.writerow([2.0, 1])

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_cli_basic_stdout(self, mock_stdout) -> None:
        test_args = ["epilink", str(self.input_path), "--mc-samples", "10"]
        with patch.object(sys, "argv", test_args):
            main()

        output = mock_stdout.getvalue()
        self.assertIn("epilink_score", output)
        self.assertEqual(len(output.strip().split("\n")), 3) # header + 2 rows

    def test_cli_output_file(self) -> None:
        test_args = ["epilink", str(self.input_path), "--output", str(self.output_path), "--mc-samples", "10"]
        with patch.object(sys, "argv", test_args):
            main()

        self.assertTrue(self.output_path.exists())
        with open(self.output_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            self.assertEqual(len(rows), 2)
            self.assertIn("epilink_score", rows[0])

    def test_cli_custom_nh_parameters(self) -> None:
        test_args = [
            "epilink", str(self.input_path),
            "--output", str(self.output_path),
            "--mc-samples", "10",
            "--incubation-shape", "6.0",
            "--substitution-rate", "0.002"
        ]
        with patch.object(sys, "argv", test_args):
            main()

        self.assertTrue(self.output_path.exists())

    @patch("sys.stderr", new_callable=io.StringIO)
    def test_cli_missing_columns(self, mock_stderr) -> None:
        invalid_input = Path(self.temp_dir.name) / "invalid.csv"
        with open(invalid_input, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["wrong_col1", "wrong_col2"])
            writer.writerow([1.0, 0])

        test_args = ["epilink", str(invalid_input)]
        with patch.object(sys, "argv", test_args), self.assertRaises(SystemExit) as cm:
            main()

        self.assertEqual(cm.exception.code, 1)

    def test_cli_invalid_argument_choice(self) -> None:
        test_args = ["epilink", str(self.input_path), "--mutation-process", "invalid"]
        with patch.object(sys, "argv", test_args), self.assertRaises(SystemExit) as cm:
            # argparse will print to stderr and exit(2)
            with patch("sys.stderr", new_callable=io.StringIO):
                main()
        self.assertEqual(cm.exception.code, 2)

    @patch("logging.getLogger")
    def test_cli_verbose_logging(self, mock_get_logger) -> None:
        # This is a bit tricky to test exactly, but we can check if it runs without error
        test_args = ["epilink", str(self.input_path), "--verbose", "--mc-samples", "10"]
        with patch.object(sys, "argv", test_args):
            main()

if __name__ == "__main__":
    unittest.main()
