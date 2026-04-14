#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
from pathlib import Path


TEST_RE = re.compile(r"test:\s*acc=(?P<acc>[0-9.]+)\s+macro_f1=(?P<f1>[0-9.]+)")
TEST_RE_ALT = re.compile(r"test\s+accuracy=(?P<acc>[0-9.]+)\s+macro_f1=(?P<f1>[0-9.]+)")


def extract_metrics(train_log: Path):
    if not train_log.exists():
        return None, None
    acc = None
    f1 = None
    for line in train_log.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = TEST_RE.search(line) or TEST_RE_ALT.search(line)
        if m:
            acc = float(m.group("acc"))
            f1 = float(m.group("f1"))
    return acc, f1


def main():
    repo_root = Path(__file__).resolve().parents[3]
    abl_root = Path(__file__).resolve().parents[1]
    exp_root = repo_root / "experiments" / "ablations_patchtst"

    rows = []
    if exp_root.exists():
        for run_dir in sorted(exp_root.glob("patchtst_*")):
            train_log = run_dir / "train.log"
            acc, f1 = extract_metrics(train_log)
            rows.append(
                {
                    "run": run_dir.name,
                    "acc": "" if acc is None else f"{acc:.4f}",
                    "macro_f1": "" if f1 is None else f"{f1:.4f}",
                    "done": "yes" if (run_dir / "DONE").exists() else "no",
                }
            )

    rows.sort(key=lambda r: (r["macro_f1"] == "", r["macro_f1"]), reverse=False)
    rows = sorted(rows, key=lambda r: float(r["macro_f1"]) if r["macro_f1"] else -1.0, reverse=True)

    csv_path = abl_root / "results" / "summary.csv"
    md_path = abl_root / "results" / "summary.md"

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["run", "acc", "macro_f1", "done"])
        w.writeheader()
        w.writerows(rows)

    lines = ["# PatchTST Ablations Summary", "", "| run | acc | macro_f1 | done |", "|---|---:|---:|---|"]
    for r in rows:
        lines.append(f"| {r['run']} | {r['acc']} | {r['macro_f1']} | {r['done']} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
