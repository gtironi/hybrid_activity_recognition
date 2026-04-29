"""Aggregate all eval/metrics_test.json from runs/ into a summary CSV.

Usage:
    python -m pretrain_ablations.results.summarize
    python -m pretrain_ablations.results.summarize --runs_dir pretrain_ablations/runs --out results/summary.csv --filter smoke_
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def collect_runs(runs_dir: Path, filter_str: str = "") -> list[dict]:
    rows = []
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        if filter_str and filter_str not in run_dir.name:
            continue
        metrics_path = run_dir / "eval" / "metrics_test.json"
        summary_path = run_dir / "eval" / "summary.txt"
        config_path  = run_dir / "artifacts" / "config.yaml"
        if not metrics_path.exists():
            continue
        with open(metrics_path) as f:
            metrics = json.load(f)

        row: dict = {"run_id": run_dir.name}

        # parse from summary.txt if available
        if summary_path.exists():
            summary = summary_path.read_text().strip()
            for part in summary.split("|"):
                part = part.strip()
                if "=" in part:
                    k, v = part.split("=", 1)
                    row[k.strip()] = v.strip()

        # fill from config.yaml if fields missing
        if config_path.exists():
            import yaml
            with open(config_path) as f:
                cfg = yaml.safe_load(f) or {}
            row.setdefault("dataset", cfg.get("data", {}).get("dataset_id", ""))
            row.setdefault("encoder", cfg.get("encoder", {}).get("name", ""))
            row.setdefault("method", cfg.get("pretext", {}).get("method", ""))
            row.setdefault("finetune", cfg.get("finetune", {}).get("mode", ""))
            row.setdefault("seed", str(cfg.get("seed", "")))

        # add metrics
        row["accuracy"]     = metrics.get("accuracy", "")
        row["macro_f1"]     = metrics.get("macro_f1", "")
        row["weighted_f1"]  = metrics.get("weighted_f1", "")
        row["precision"]    = metrics.get("precision_macro", "")
        row["recall"]       = metrics.get("recall_macro", "")
        per_class = metrics.get("per_class_f1", {})
        for cls, val in per_class.items():
            row[f"f1_{cls}"] = val
        rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", default="pretrain_ablations/runs")
    p.add_argument("--out", default="pretrain_ablations/results/summary.csv")
    p.add_argument("--filter", default="", help="Only include runs whose name contains this string")
    args = p.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        print(f"No runs dir: {runs_dir}")
        return

    rows = collect_runs(runs_dir, args.filter)
    if not rows:
        print("No completed runs found.")
        return

    all_keys = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                all_keys.append(k); seen.add(k)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} runs to {out_path}")
    # print table
    print(f"\n{'run_id':40s} {'encoder':12s} {'method':12s} {'finetune':12s} {'acc':6s} {'macro_f1':8s}")
    print("-" * 100)
    for row in rows:
        print(f"{row.get('run_id','')[:40]:40s} "
              f"{str(row.get('encoder',''))[:12]:12s} "
              f"{str(row.get('method',''))[:12]:12s} "
              f"{str(row.get('finetune',''))[:12]:12s} "
              f"{str(row.get('accuracy',''))[:6]:6s} "
              f"{str(row.get('macro_f1',''))[:8]:8s}")


if __name__ == "__main__":
    main()
