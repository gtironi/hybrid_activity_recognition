"""Generate a markdown table with overall metrics for every experiment.

For each split directory inside ``experiments/`` (e.g. ``0.2_val``, ``0.5_val``),
this script collects ``overall`` metrics from each model's
``test_classification_metrics.json`` and writes a ``summary.md`` table inside
that split directory. It also writes a combined ``summary.md`` at the
``experiments/`` root that aggregates every split.
"""

from __future__ import annotations

import json
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1] / "experiments"
METRICS_FILENAME = "test_classification_metrics.json"
SUMMARY_FILENAME = "summary.md"


def load_overall(metrics_path: Path) -> dict[str, float] | None:
    try:
        with metrics_path.open("r") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return data.get("overall")


def collect_split(split_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        metrics_path = model_dir / METRICS_FILENAME
        if not metrics_path.is_file():
            continue
        overall = load_overall(metrics_path)
        if overall is None:
            continue
        rows.append(
            {
                "model": model_dir.name,
                "accuracy": overall.get("accuracy"),
                "f1_macro": overall.get("f1_macro"),
                "f1_weighted": overall.get("f1_weighted"),
            }
        )
    return rows


def format_value(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return "-"


def render_table(rows: list[dict[str, object]]) -> str:
    rows_sorted = sorted(
        rows,
        key=lambda r: r["f1_macro"] if isinstance(r["f1_macro"], (int, float)) else -1,
        reverse=True,
    )
    header = "| Model | Accuracy | F1 Macro | F1 Weighted |\n"
    sep = "|---|---|---|---|\n"
    body = "".join(
        f"| {r['model']} | {format_value(r['accuracy'])} "
        f"| {format_value(r['f1_macro'])} | {format_value(r['f1_weighted'])} |\n"
        for r in rows_sorted
    )
    return header + sep + body


def main() -> None:
    if not EXPERIMENTS_DIR.is_dir():
        raise SystemExit(f"experiments directory not found: {EXPERIMENTS_DIR}")

    combined_sections: list[str] = []
    for split_dir in sorted(p for p in EXPERIMENTS_DIR.iterdir() if p.is_dir()):
        rows = collect_split(split_dir)
        if not rows:
            print(f"[skip] no metrics found in {split_dir.name}")
            continue

        table_md = render_table(rows)
        split_summary = f"# Summary: {split_dir.name}\n\n{table_md}"
        (split_dir / SUMMARY_FILENAME).write_text(split_summary)
        print(f"[ok] wrote {split_dir / SUMMARY_FILENAME} ({len(rows)} models)")

        combined_sections.append(f"## {split_dir.name}\n\n{table_md}")

    if combined_sections:
        combined = "# Experiments summary\n\n" + "\n".join(combined_sections)
        (EXPERIMENTS_DIR / SUMMARY_FILENAME).write_text(combined)
        print(f"[ok] wrote {EXPERIMENTS_DIR / SUMMARY_FILENAME}")


if __name__ == "__main__":
    main()
