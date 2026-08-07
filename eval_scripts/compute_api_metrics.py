#!/usr/bin/env python3
"""Compute BERT-Sim / BLEU / ROUGE metrics for API baseline JSON files.

   Supports our nested output format (one entry per image, vqa/report inside):

   VQA input (from api_baseline_generate.py):
       [{"image_id": "xxx", "vqa": [{"question": "...", "gt_answer": "...",
          "pred_answer": "..." }]}]

   Report input:
       [{"image_id": "xxx", "report": {"gt_answer": "...", "pred_answer": "..."}, ...}]

   Usage:
       python eval_scripts/compute_api_metrics.py \
           --pred results/kvasir_vqa_qwen3vl.json \
           --task vqa \
           --dataset kvasir
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ===========================================================================
# Argument parsing
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate API baseline outputs")
    p.add_argument(
        "--pred", type=str, required=True, nargs="+",
        help="One or more prediction JSON files from api_baseline_generate.py",
    )
    p.add_argument(
        "--task", type=str, required=True, choices=["vqa", "report"],
        help="Task: vqa or report",
    )
    p.add_argument(
        "--dataset", type=str, default="kvasir",
        help="Dataset name (for logging)",
    )
    p.add_argument(
        "--output-dir", type=str, default="results/metrics",
        help="Directory for CSV outputs (default: results/metrics)",
    )
    p.add_argument(
        "--model-name", type=str, default="bert-sim",
        help="BERT model for similarity (default: paraphrase-MiniLM-L6-v2, fast/cpu-friendly)",
    )
    return p.parse_args()


# ===========================================================================
# BERT similarity (lazy-load for server environments)
# ===========================================================================

_bert_model = None


def _get_bert_model(model_name: str):
    global _bert_model
    if _bert_model is None:
        from sentence_transformers import SentenceTransformer
        logging.info("Loading BERT model: %s", model_name)
        _bert_model = SentenceTransformer(model_name)
    return _bert_model


def bert_sim(text_a: str, text_b: str, model_name: str) -> float:
    model = _get_bert_model(model_name)
    from sentence_transformers import util
    with torch.no_grad():
        pass  # SentenceTransformer handles this internally
    emb_a = model.encode([str(text_a)], convert_to_tensor=True)
    emb_b = model.encode([str(text_b)], convert_to_tensor=True)
    return float(util.pytorch_cos_sim(emb_a, emb_b)[0][0].item())


# ===========================================================================
# BLEU-4
# ===========================================================================

import math
from collections import Counter


def _ngram_counts(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _mod_precision(pred, gt, n):
    pc = _ngram_counts(pred, n)
    gc = _ngram_counts(gt, n)
    if not pc:
        return 0.0
    clipped = sum(min(c, gc.get(ng, 0)) for ng, c in pc.items())
    return clipped / sum(pc.values())


def bleu4(pred: str, gt: str) -> float:
    pt = pred.lower().split()
    gt_t = gt.lower().split()
    if not pt or not gt_t:
        return 0.0
    precs = [_mod_precision(pt, gt_t, n) or 1e-7 for n in range(1, 5)]
    geo = math.exp(sum(math.log(p) for p in precs) / 4)
    bp = 1.0 if len(pt) > len(gt_t) else math.exp(1 - len(gt_t) / max(len(pt), 1))
    return bp * geo


# ===========================================================================
# ROUGE-L
# ===========================================================================

def _lcs_len(a, b):
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if a[i - 1] == b[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def rouge_l(pred: str, gt: str) -> float:
    pt = pred.lower().split()
    gt_t = gt.lower().split()
    if not pt or not gt_t:
        return 0.0
    lcs = _lcs_len(pt, gt_t)
    prec = lcs / len(pt) if pt else 0.0
    rec = lcs / len(gt_t) if gt_t else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


# ===========================================================================
# Data loading — handles nested format from api_baseline_generate.py
# ===========================================================================

def load_vqa_pairs(pred_path: str) -> list[dict]:
    """Load VQA predictions from nested or flat JSON format.
    Returns flat list: [{"image_id": ..., "question": ..., "gt": ..., "pred": ...}, ...]
    """
    with open(pred_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pairs = []
    for item in data:
        img_id = item.get("image_id", "")
        # Nested: {"image_id":"...", "vqa":[{question,gt_answer,pred_answer},...]}
        if "vqa" in item:
            for qa in item["vqa"]:
                pairs.append({
                    "image_id": img_id,
                    "question": qa.get("question", ""),
                    "gt": str(qa.get("gt_answer", "")),
                    "pred": str(qa.get("pred_answer", "")),
                })
        # Flat: {"image_id":"...", "question":"...", "gt_answer":"...", "pred_answer":"..."}
        elif "question" in item and "gt_answer" in item:
            pairs.append({
                "image_id": img_id,
                "question": item.get("question", ""),
                "gt": str(item.get("gt_answer", "")),
                "pred": str(item.get("pred_answer", "")),
            })
    return pairs


def load_report_pairs(pred_path: str) -> list[dict]:
    """Load report predictions from nested or flat JSON format.
    Returns flat list: [{"image_id": ..., "gt": ..., "pred": ...}, ...]
    """
    with open(pred_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pairs = []
    for item in data:
        img_id = item.get("image_id", "")
        if "report" in item:
            r = item["report"]
            pairs.append({
                "image_id": img_id,
                "gt": str(r.get("gt_answer", "")),
                "pred": str(r.get("pred_answer", "")),
            })
        elif "gt_answer" in item:
            pairs.append({
                "image_id": img_id,
                "gt": str(item.get("gt_answer", "")),
                "pred": str(item.get("pred_answer", "")),
            })
    return pairs


# ===========================================================================
# Evaluation functions
# ===========================================================================

def evaluate_vqa(pred_path: str, dataset: str, output_dir: str, model_name: str) -> dict:
    pairs = load_vqa_pairs(pred_path)
    if not pairs:
        logging.error("No VQA pairs found in %s", pred_path)
        return {}

    stem = Path(pred_path).stem
    scores = []
    rows = []
    for p in pairs:
        s = bert_sim(p["pred"], p["gt"], model_name)
        scores.append(s)
        rows.append({"image_id": p["image_id"], "question": p["question"][:80],
                     "gt": p["gt"][:80], "pred": p["pred"][:80], "BERT_score": s})

    avg = sum(scores) / len(scores) if scores else 0.0

    csv_path = os.path.join(output_dir, f"{stem}_vqa_bert.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    logging.info("VQA BERT-Sim for %s (%s): %.4f  ->  %s", dataset, stem, avg, csv_path)

    return {"bert_sim": avg, "n_samples": len(pairs)}


def evaluate_report(pred_path: str, dataset: str, output_dir: str, model_name: str) -> dict:
    pairs = load_report_pairs(pred_path)
    if not pairs:
        logging.error("No report pairs found in %s", pred_path)
        return {}

    stem = Path(pred_path).stem
    bert_scores, bleu_scores, rouge_scores = [], [], []
    rows = []
    for p in pairs:
        b = bert_sim(p["pred"], p["gt"], model_name)
        bl = bleu4(p["pred"], p["gt"])
        rl = rouge_l(p["pred"], p["gt"])
        bert_scores.append(b); bleu_scores.append(bl); rouge_scores.append(rl)
        rows.append({"image_id": p["image_id"], "BERT_score": b, "BLEU_4": bl, "ROUGE_L": rl,
                     "gt": p["gt"][:100], "pred": p["pred"][:100]})

    n = len(pairs)
    avg_b = sum(bert_scores) / n if n else 0.0
    avg_bl = sum(bleu_scores) / n if n else 0.0
    avg_rl = sum(rouge_scores) / n if n else 0.0

    csv_path = os.path.join(output_dir, f"{stem}_report_metrics.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    logging.info("Report metrics for %s (%s): BERT-Sim=%.4f  BLEU-4=%.4f  ROUGE-L=%.4f  ->  %s",
                 dataset, stem, avg_b, avg_bl, avg_rl, csv_path)

    result = {"bert_sim": avg_b, "bleu4": avg_bl, "rouge_l": avg_rl, "n_samples": n}

    # CheXbert-F1 — only for chest X-ray (indiana/rsna)
    if dataset in ("indiana", "rsna", "chest"):
        tmp_dir = os.path.join(output_dir, "_tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        gt_path = os.path.join(tmp_dir, f"{stem}_gt.json")
        pred_tmp = os.path.join(tmp_dir, f"{stem}_pred_flat.json")
        gt_flat = [{"image_id": p["image_id"], "caption": p["gt"]} for p in pairs]
        pred_flat = [{"image_id": p["image_id"], "caption": p["pred"]} for p in pairs]
        with open(gt_path, "w") as f: json.dump(gt_flat, f)
        with open(pred_tmp, "w") as f: json.dump(pred_flat, f)
        try:
            from eval_scripts.metrics import chexbert_f1
            chex_csv = os.path.join(output_dir, f"{stem}_chexbert_f1.csv")
            chex = chexbert_f1(gt_path, pred_tmp, dataset, chex_csv)
            result["chexbert_micro_f1"] = chex.get("micro_f1", 0.0)
            result["chexbert_macro_f1"] = chex.get("macro_f1", 0.0)
            logging.info("CheXbert-F1 for %s (%s): micro=%.4f macro=%.4f",
                         dataset, stem, result["chexbert_micro_f1"], result["chexbert_macro_f1"])
        except Exception as e:
            logging.warning("CheXbert-F1 skipped: %s", e)

    return result


# ===========================================================================
# Main
# ===========================================================================

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Task: {args.task.upper()}  |  Dataset: {args.dataset}")
    print(f"{'='*70}")

    summary = []
    for pred_path in args.pred:
        if not os.path.exists(pred_path):
            logging.warning("File not found: %s", pred_path)
            continue

        if args.task == "vqa":
            r = evaluate_vqa(pred_path, args.dataset, args.output_dir, args.model_name)
        else:
            r = evaluate_report(pred_path, args.dataset, args.output_dir, args.model_name)

        stem = Path(pred_path).stem
        if r:
            r["model"] = stem
            summary.append(r)

    # Print comparison table
    if len(summary) > 1:
        print(f"\n{'='*70}")
        print(f"  Comparison Summary")
        print(f"{'='*70}")
        if args.task == "vqa":
            print(f"  {'Model':<35s} {'BERT-Sim':>10s}  {'#Samples':>10s}")
            print(f"  {'-'*57}")
            for s in summary:
                print(f"  {s['model']:<35s} {s['bert_sim']:>10.4f}  {s.get('n_samples', 0):>10d}")
        else:
            has_chex = any("chexbert" in k for k in summary[0])
            if has_chex:
                print(f"  {'Model':<35s} {'BERT':>8s} {'BLEU4':>8s} {'ROUGE':>8s} {'CFx_m':>8s} {'CFx_M':>8s} {'#':>6s}")
                print(f"  {'-'*85}")
                for s in summary:
                    print(f"  {s['model']:<35s} {s['bert_sim']:>8.4f} {s['bleu4']:>8.4f} {s['rouge_l']:>8.4f} {s.get('chexbert_micro_f1',0):>8.4f} {s.get('chexbert_macro_f1',0):>8.4f} {s.get('n_samples',0):>6d}")
            else:
                print(f"  {'Model':<35s} {'BERT-Sim':>10s} {'BLEU-4':>10s} {'ROUGE-L':>10s}  {'#Samples':>10s}")
                print(f"  {'-'*77}")
                for s in summary:
                    print(f"  {s['model']:<35s} {s['bert_sim']:>10.4f} {s['bleu4']:>10.4f} {s['rouge_l']:>10.4f}  {s.get('n_samples', 0):>10d}")
        print(f"{'='*70}\n")

    # Save summary JSON
    summary_path = os.path.join(args.output_dir, f"{args.dataset}_{args.task}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logging.info("Summary saved to %s", summary_path)


if __name__ == "__main__":
    import torch
    main()
