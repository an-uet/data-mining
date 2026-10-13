#!/usr/bin/env python
"""Run the complete Beijing PM2.5 (Aotizhongxin) HW1 experiment.

    python run_all.py                 # everything, in protocol order (~20 min on 8 cores)
    python run_all.py --until freeze  # development: stop before the test set is opened
    python run_all.py --data-path /path/to/PRSA_Data_Aotizhongxin_20130301-20170228.csv

Outputs go to results/{figures,tables,models,logs}; a concise summary is printed at the end and saved to results/final_summary.txt.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.experiment import Experiment  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-path", default=None, help="raw CSV (default: data/raw/<file>, downloaded from the UCI archive if missing)")
    ap.add_argument("--until", choices=["freeze"], default=None, help="stop before the held-out test set is opened")
    args = ap.parse_args()
    Experiment(data_path=args.data_path).run_all(until=args.until)


if __name__ == "__main__":
    main()
