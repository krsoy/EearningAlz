#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="prediction_model_outputs")
    p.add_argument("--test-start-quarter", default="2024Q1")
    p.add_argument("--write-csv-copy", action="store_true")
    p.add_argument("--run-random-forest", action="store_true")
    p.add_argument("--community-file", default="", help="HF community file or local path")
    p.add_argument("--community-local-file", default="", help="Local cluster assignment CSV/Parquet, e.g. best_company_cluster_assignment.csv")
    p.add_argument("--community-dataset", default="", help="Optional HF dataset containing the community file")
    p.add_argument("--same-quarter-include-unordered", action="store_true")
    return p.parse_args()


def run(cmd):
    print("\n" + "=" * 90)
    print("RUN:", " ".join(cmd))
    print("=" * 90)
    subprocess.run(cmd, check=True)


def main():
    args = parse_args(); out_dir = Path(args.out_dir)
    build = [sys.executable, "01_build_prediction_dataset.py", "--out-dir", str(out_dir)]
    if args.write_csv_copy: build.append("--write-csv-copy")
    if args.community_file: build += ["--community-file", args.community_file]
    if args.community_local_file: build += ["--community-local-file", args.community_local_file]
    if args.community_dataset: build += ["--community-dataset", args.community_dataset]
    if args.same_quarter_include_unordered: build.append("--same-quarter-include-unordered")
    train = [sys.executable, "02_train_prediction_models.py", "--input-dir", str(out_dir), "--out-dir", str(out_dir / "model_results"), "--test-start-quarter", args.test_start_quarter, "--dataset", "both"]
    if args.run_random_forest: train.append("--run-random-forest")
    run(build); run(train)
    print("\nDONE")
    print("Main outputs:")
    print(" ", out_dir / "prediction_dataset_cross_quarter.parquet")
    print(" ", out_dir / "prediction_dataset_same_quarter_ordered.parquet")
    print(" ", out_dir / "model_results" / "prediction_model_metrics.csv")

if __name__ == "__main__": main()
