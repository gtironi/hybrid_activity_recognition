"""Aggregate all rf_hypersearch comparison.json under a base dir into one table.

Builds SUMMARY.md + summary.csv with default-vs-tuned test metrics and the
tuning gain (tuned_trainval - default_trainval) for every dataset/fold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

MET = ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    args = ap.parse_args()
    base = Path(args.base)

    rows = []
    for cj in sorted(base.rglob("comparison.json")):
        d = json.loads(cj.read_text())
        label = cj.parent.relative_to(base).as_posix()
        tr = d.get("test_results", {})
        rec = {"dataset": label, "select_metric": d.get("select_metric"),
               "best_params": json.dumps(d.get("best_params"))}
        for model in ["default_tr", "default_trainval", "tuned_tr", "tuned_trainval"]:
            for m in MET:
                rec[f"{model}.{m}"] = tr.get(model, {}).get(m)
        # headline gain: tuned vs default, both on train+val (production-comparable)
        sm = d.get("select_metric", "balanced_accuracy")
        dft = tr.get("default_trainval", {}).get(sm)
        tnd = tr.get("tuned_trainval", {}).get(sm)
        rec["gain_trainval_" + sm] = (tnd - dft) if (dft is not None and tnd is not None) else None
        rows.append(rec)

    if not rows:
        print("No comparison.json found under", base)
        return
    df = pd.DataFrame(rows).sort_values("dataset")
    df.to_csv(base / "summary.csv", index=False)

    sm_col = [c for c in df.columns if c.startswith("gain_trainval_")][0]
    sm = sm_col.replace("gain_trainval_", "")
    lines = [
        f"# RF hypersearch — all datasets ({len(df)} runs)",
        "",
        f"Selection metric: **{sm}**. Gain = tuned_trainval − default_trainval ({sm}).",
        "Metrics are test-set %; `default_trainval` = current production RF.",
        "",
        f"| Dataset | default {sm} | tuned {sm} | **gain (pp)** | default f1m | tuned f1m |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        g = r[sm_col]
        lines.append(
            f"| {r['dataset']} "
            f"| {r[f'default_trainval.{sm}']*100:.2f} "
            f"| {r[f'tuned_trainval.{sm}']*100:.2f} "
            f"| {'' if g is None else f'{g*100:+.2f}'} "
            f"| {r['default_trainval.f1_macro']*100:.2f} "
            f"| {r['tuned_trainval.f1_macro']*100:.2f} |"
        )
    valid = df[sm_col].dropna()
    if len(valid):
        lines += ["", f"**Mean gain: {valid.mean()*100:+.2f} pp** · "
                      f"helps in {(valid > 0).sum()}/{len(valid)} datasets."]
    (base / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {base/'SUMMARY.md'} and {base/'summary.csv'}  ({len(df)} runs)")


if __name__ == "__main__":
    main()
