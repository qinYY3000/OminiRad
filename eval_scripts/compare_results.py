"""Compare evaluation results across models and generate summary tables.

Scans ``eval_results/`` for all ``summary.json`` files and produces
comparison CSV tables under ``eval_results/comparison_tables/``.

Usage
-----
    python eval_scripts/compare_results.py
    # → reads eval_results/*/*/summary.json
    # → writes eval_results/comparison_tables/table{1-5}.csv

    # Custom root directory
    python eval_scripts/compare_results.py --root eval_results --output eval_results/comparison_tables
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd


# ---------------------------------------------------------------------------
# Task → table mapping
# ---------------------------------------------------------------------------
# Each table groups related tasks and their metrics.

TABLES = {
    "table1_report_generation": {
        "title": "Table 1: Report Generation (Indiana CXR)",
        "datasets": ["indiana_cxr"],
        "metrics": ["bert_sim", "bleu4", "rouge_l", "chexbert_micro_f1", "chexbert_macro_f1"],
        "metric_labels": {
            "bert_sim": "BERT-Sim",
            "bleu4": "BLEU-4",
            "rouge_l": "ROUGE-L",
            "chexbert_micro_f1": "CheXbert micro-F1",
            "chexbert_macro_f1": "CheXbert macro-F1",
        },
    },
    "table2_vqa": {
        "title": "Table 2: Visual Question Answering",
        "datasets": ["radvqa", "slake_vqa"],
        "metrics": ["bert_sim"],
        "metric_labels": {"bert_sim": "BERT-Sim"},
    },
    "table3_detection_grounding": {
        "title": "Table 3: Detection & Grounding",
        "datasets": ["rsna", "SLAKE"],
        "metrics": ["iou"],
        "metric_labels": {"iou": "IoU"},
    },
    "table4_ultrasound_multitask": {
        "title": "Table 4: Ultrasound Multi-task (Group-Breast / Group-Thyroid US)",
        "datasets": ["group_breast_us", "group_thyroid_us"],
        "metrics": [
            "segmentation_dice", "detection_iou", "report_bert_sim",
            "refer_iou", "identify_accuracy",
        ],
        "metric_labels": {
            "segmentation_dice": "Dice",
            "detection_iou": "Det-IoU",
            "report_bert_sim": "Report-BERT-Sim",
            "refer_iou": "Refer-IoU",
            "identify_accuracy": "Identify-Acc",
        },
    },
    "table5_ablation": {
        "title": "Table 5: Ablation Study (OmniRad variants)",
        "datasets": ["indiana_cxr", "rsna", "group_breast_us"],
        "metrics": ["bert_sim", "iou", "segmentation_dice"],
        "metric_labels": {
            "bert_sim": "Report-BERT-Sim",
            "iou": "Det-IoU",
            "segmentation_dice": "Dice",
        },
        # Only include models that start with "omnirad" (ablation variants)
        "model_filter": lambda name: name.startswith("omnirad"),
    },
}


# ---------------------------------------------------------------------------
# Harvesting
# ---------------------------------------------------------------------------
def collect_summaries(root: str) -> dict:
    """Walk ``root`` and collect all ``summary.json`` files.

    Returns
    -------
    dict: {model_name: {task_name: summary_dict}}
    """
    results: dict = {}
    root = Path(root)

    if not root.exists():
        print(f"[compare] root directory {root} does not exist")
        return results

    for summary_path in sorted(root.rglob("summary.json")):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  [warn] cannot read {summary_path}: {e}")
            continue

        model = summary.get("model", "unknown")
        task = summary.get("task", summary_path.parent.name)
        metrics = summary.get("metrics", {})

        results.setdefault(model, {})[task] = {
            "metrics": metrics,
            "n_samples": summary.get("n_samples", 0),
            "path": str(summary_path),
        }

    return results


def build_table(table_config: dict, all_results: dict) -> pd.DataFrame:
    """Build a comparison DataFrame for one table."""
    rows = []
    datasets = table_config["datasets"]
    metrics = table_config["metrics"]
    model_filter = table_config.get("model_filter")

    for model_name, tasks in sorted(all_results.items()):
        if model_filter and not model_filter(model_name):
            continue

        row = {"Model": model_name}
        for dataset in datasets:
            task_data = tasks.get(dataset)
            if task_data is None:
                # Try with _report / _detection suffix (ultrasound)
                for suffix in ["_report", "_segmentation", "_detection",
                               "_refer", "_identify"]:
                    task_data = tasks.get(f"{dataset}{suffix}")
                    if task_data:
                        break

            if task_data is None:
                for m in metrics:
                    col = f"{dataset}_{m}"
                    row[col] = "-"
                continue

            task_metrics = task_data["metrics"]
            for m in metrics:
                col = f"{dataset}_{m}"
                val = task_metrics.get(m)
                if val is not None:
                    row[col] = round(val, 4)
                else:
                    row[col] = "-"
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Reorder columns: Model first, then by dataset
    cols = ["Model"]
    for dataset in datasets:
        for m in metrics:
            col = f"{dataset}_{m}"
            if col in df.columns:
                cols.append(col)
    df = df[[c for c in cols if c in df.columns]]

    return df


def build_ultrasound_table(all_results: dict) -> pd.DataFrame:
    """Special table for ultrasound multi-task results.

    Ultrasound models have sub-task summaries stored as:
      group_breast_us/report/summary.json
      group_breast_us/segmentation/summary.json
      ...
    """
    rows = []
    us_datasets = ["group_breast_us", "group_thyroid_us"]
    sub_tasks = ["report", "segmentation", "detection", "refer", "identify"]
    metric_map = {
        "report": "bert_sim",
        "segmentation": "dice",
        "detection": "iou",
        "refer": "iou",
        "identify": "accuracy",
    }

    for model_name, tasks in sorted(all_results.items()):
        row = {"Model": model_name}
        for ds in us_datasets:
            for sub in sub_tasks:
                full_task = f"{ds}/{sub}"
                # Look for this as a nested path
                task_data = None
                for key in [full_task, f"{ds}_{sub}", ds]:
                    if key in tasks:
                        task_data = tasks[key]
                        break

                if task_data is None:
                    # Try direct lookup with underscore convention
                    for key in tasks:
                        if ds in key and sub in key:
                            task_data = tasks[key]
                            break

                col = f"{ds}_{sub}"
                if task_data:
                    metric_key = metric_map.get(sub, sub)
                    val = task_data["metrics"].get(metric_key)
                    if val is None:
                        # Try alternative metric names
                        for alt_key in ["bert_sim", "iou", "dice", "accuracy"]:
                            val = task_data["metrics"].get(alt_key)
                            if val is not None:
                                break
                    row[col] = round(val, 4) if val is not None else "-"
                else:
                    row[col] = "-"
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=str, default="eval_results",
                        help="Root directory containing model subdirectories.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for CSV tables (default: <root>/comparison_tables).")
    args = parser.parse_args()

    output_dir = args.output or os.path.join(args.root, "comparison_tables")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Scanning {args.root} for summary.json files...")
    all_results = collect_summaries(args.root)

    if not all_results:
        print("No results found. Run evaluations first.")
        return

    n_models = len(all_results)
    n_tasks = sum(len(v) for v in all_results.values())
    print(f"Found {n_models} models × {n_tasks} task results.\n")

    for model, tasks in sorted(all_results.items()):
        print(f"  {model}: {list(tasks.keys())}")

    # Build standard tables
    for table_name, table_config in TABLES.items():
        if table_name == "table4_ultrasound_multitask":
            df = build_ultrasound_table(all_results)
        else:
            df = build_table(table_config, all_results)

        if df.empty:
            print(f"\n[{table_name}] No data — skipped.")
            continue

        csv_path = os.path.join(output_dir, f"{table_name}.csv")
        df.to_csv(csv_path, index=False)

        print(f"\n{'='*70}")
        print(f"  {table_config['title']}")
        print(f"{'='*70}")
        print(df.to_string(index=False))
        print(f"\n  → saved to {csv_path}")

    # Also write a combined flat CSV
    flat_rows = []
    for model, tasks in sorted(all_results.items()):
        for task, data in tasks.items():
            row = {"Model": model, "Task": task}
            row.update(data["metrics"])
            row["n_samples"] = data.get("n_samples", 0)
            flat_rows.append(row)
    if flat_rows:
        flat_df = pd.DataFrame(flat_rows)
        flat_csv = os.path.join(output_dir, "all_results_flat.csv")
        flat_df.to_csv(flat_csv, index=False)
        print(f"\n  → flat results saved to {flat_csv}")

    print(f"\n{'='*70}")
    print(f"All comparison tables saved to: {output_dir}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
