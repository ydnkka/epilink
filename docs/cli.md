# CLI

The `epilink` command provides quick access to the estimators from the shell.

Examples:

- One pair:
  epilink point -g 2 -t 4 --nsims 10000

- Multiple pairs (CSV):
  epilink point -g 0 1 2 -t 0 2 5 --nsims 500 > out.csv

- Grid:
  epilink grid --g-start 0 --g-stop 5 --g-step 1 --t-start 0 --t-stop 12 --t-step 3 --out grid.csv
