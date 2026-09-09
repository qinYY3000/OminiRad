import sys
sys.path.append('.')

import json
import os
import re
import csv
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from minigpt4.common.eval_utils import computeIoU

# Load pre-trained BERT model
# Priority: 1) BERT_MODEL_PATH env var  2) default local path
_BERT_PATH = os.environ.get("BERT_MODEL_PATH") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "weights", "bert_model"
)
_BERT_PATH = os.path.abspath(_BERT_PATH)

if not os.path.exists(os.path.join(_BERT_PATH, "modules.json")):
    print(f"[metrics] BERT model not found at {_BERT_PATH}")
    print(f"[metrics] Please download paraphrase-MiniLM-L6-v2 and place it at:")
    print(f"[metrics]   {_BERT_PATH}")
    print(f"[metrics] Or set BERT_MODEL_PATH env var to the correct path.")
    raise FileNotFoundError(f"BERT model not found at {_BERT_PATH}")

model = SentenceTransformer(str(_BERT_PATH))


# --------------------------------------------------------------------------- #
#  Helper: normalise image_ids so matching is robust to file extensions
# --------------------------------------------------------------------------- #
def _norm_id(image_id: str) -> str:
    """Strip **all** file extensions so both ``foo.dcm.png`` and ``foo.dcm``
    become ``"foo"``, making ID matching independent of how extensions are
    handled by upstream cleaning steps."""
    while True:
        root, ext = os.path.splitext(image_id)
        if not ext:
            return root
        image_id = root


# BERT similarity function will be utilized in the two following functions
def compute_bert_similarity(prediction_caption, ground_truth_caption):
    prediction_embedding = model.encode([prediction_caption])
    ground_truth_embedding = model.encode([ground_truth_caption])
    similarity = util.pytorch_cos_sim(prediction_embedding, ground_truth_embedding)[0][0].item()
    return similarity


def report_bert_sim(gt_pth, pred_pth, output_csv):
    """BERT-similarity between ground-truth captions and model predictions.

    Used for any report-generation dataset whose ground truth is laid out as::

        [{"image_id": "...", "caption": "..."}, ...]

    and whose predictions have been normalised by ``clean_report_json``.
    """
    # Read the ground truth and prediction JSON files
    with open(gt_pth, 'r') as f:
        ground_truth_data = json.load(f)

    with open(pred_pth, 'r') as f:
        prediction_data = json.load(f)

    # Build a lookup keyed by normalised image_id (no extensions) for robust
    # matching regardless of whether IDs carry .dcm, .png, both, or neither.
    gt_by_id = {}
    for gt_item in ground_truth_data:
        # Accept "caption" (report-gen datasets) or "answer" (multi-task unified)
        gt_text = gt_item.get("caption") or gt_item.get("answer") or ""
        gt_by_id[_norm_id(gt_item["image_id"])] = gt_text

    # Create a list to store BERT similarity data
    bert_similarity_data = []

    # Initialize variables to calculate the average
    total_similarity = 0
    total_count = 0

    # Iterate over each item in the prediction_data list
    for item in prediction_data:
        # Extract the image_id and corresponding prediction caption
        image_id = item["image_id"]
        prediction_caption = item.get("caption") or item.get("answer") or ""

        # Match by normalised id (strip .dcm / .png / both)
        ground_truth_caption = gt_by_id.get(_norm_id(image_id))

        if ground_truth_caption is not None:
            bert_similarity = compute_bert_similarity(prediction_caption, ground_truth_caption)
            bert_similarity_data.append({"image_id": image_id, "BERT_score": bert_similarity})

            total_similarity += bert_similarity
            total_count += 1
    
    average_similarity = total_similarity / total_count if total_count > 0 else 0

    if not bert_similarity_data:
        print(f"[report_bert_sim] WARNING: No matching image_ids found!")
        print(f"  Prediction has {len(prediction_data)} items, "
              f"ground truth has {len(ground_truth_data)} items.")
        print(f"  First 3 pred image_ids: "
              f"{[item.get('image_id', '?') for item in prediction_data[:3]]}")
        print(f"  First 3 GT   image_ids: "
              f"{[item.get('image_id', '?') for item in ground_truth_data[:3]]}")
        pd.DataFrame(columns=["image_id", "BERT_score"]).to_csv(output_csv, index=False)
        return average_similarity

    df = pd.DataFrame(bert_similarity_data)
    df_sorted = df.sort_values(by="BERT_score", ascending=True)
    df_sorted.to_csv(output_csv, index=False)

    return average_similarity

def VQA_BERT_Sim(gt_pth, pred_pth, output_csv):
    # Load ground truth JSON file
    with open(gt_pth, 'r') as file:
        gt_data = json.load(file)

    # Load prediction JSON file
    with open(pred_pth, 'r') as file:
        prediction_data = json.load(file)

    # RadVQA uses "image_name", SLAKE-VQA uses "img_name" — auto-detect
    img_key = "image_name"  # default
    if gt_data and isinstance(gt_data, list):
        first = gt_data[0]
        if "img_name" in first:
            img_key = "img_name"

    gt_qa_pairs = {(entry[img_key], entry['question']): entry['answer'] for entry in gt_data}

    def convert_to_dict(data):
        qa_dict = {}
        for image_name, qa_list in data.items():
            for qa in qa_list:
                key = (image_name, qa['question'])
                qa_dict[key] = qa['answer']
        return qa_dict

    pred_qa_dict = convert_to_dict(prediction_data)

    # Compute BERT similarity and create a list of results
    results = []

    for key, gt_answer in gt_qa_pairs.items():
        if key in pred_qa_dict:
            pred_answer = pred_qa_dict[key]
            gt_answer = str(gt_answer)
            pred_answer = str(pred_answer)

            # Compute BERT similarity
            similarity_score = compute_bert_similarity(pred_answer, gt_answer)

            # Append the result to the list
            results.append({
                "img_name": key[0],
                "question": key[1],
                "answer": pred_answer,
                "BERT_score": similarity_score
            })

    average_similarity = sum(entry["BERT_score"] for entry in results) / len(results) if results else 0

    if not results:
        print(f"[VQA_BERT_Sim] WARNING: No matching QA pairs found!")
        print(f"  GT has {len(gt_qa_pairs)} QA pairs, "
              f"pred has {len(pred_qa_dict)} QA pairs.")
        pd.DataFrame(columns=["img_name", "question", "answer", "BERT_score"]
                     ).to_csv(output_csv, index=False)
    else:
        df = pd.DataFrame(results)
        df_sorted = df.sort_values(by="BERT_score", ascending=True)
        df_sorted.to_csv(output_csv, index=False)

    print(f"Average BERT similarity score: {average_similarity}")
    return average_similarity


def _get_gt_key(gt_item):
    """Return the image identifier from a ground-truth item, regardless of schema."""
    return gt_item.get("key") or gt_item.get("folder_name") or gt_item.get("image_id") or ""


def _get_gt_bboxes(gt_item):
    """Return a list of [x1,y1,x2,y2] bboxes from a ground-truth item.

    Handles RSNA flat ``bbox`` field and SLAKE nested ``detection`` field.
    """
    # RSNA / unified schema: flat list of bboxes
    if "bbox" in gt_item:
        return gt_item["bbox"]

    # SLAKE schema: detection = [{"Disease": [x1,y1,x2,y2]}, ...]
    detection = gt_item.get("detection", [])
    if detection:
        bboxes = []
        for d in detection:
            for disease_name, coords in d.items():
                bboxes.append(coords)
        return bboxes

    return []


def _get_gt_size(gt_item):
    """Return (height, width) from a ground-truth item."""
    if "image_size" in gt_item:
        sz = gt_item["image_size"]
        return sz.get("height", 0), sz.get("width", 0)
    return gt_item.get("height", 0), gt_item.get("width", 0)


#################################
##############IoU################
#################################

def preprocess_bbox(bbox, original_size, image_size):
    x1 = int((bbox[0] / original_size) * image_size)
    y1 = int((bbox[1] / original_size) * image_size)
    x2 = int((bbox[2] / original_size) * image_size)
    y2 = int((bbox[3] / original_size) * image_size)
    return [x1, y1, x2, y2]

def average_iou(gt_pth, pred_pth, original_size, image_size, dataset_name, csv_filename):
    # Load ground truth
    with open(gt_pth, 'r') as file:
        ground_truth = json.load(file)

    # Load predictions
    with open(pred_pth, 'r') as file:
        predictions = json.load(file)

    # Index predictions by normalised key (no extensions)
    pred_by_key = {}
    for pred_item in predictions:
        raw_key = pred_item.get("key", "")
        pred_by_key[_norm_id(raw_key)] = pred_item.get("bbox", [])

    iou_list = []

    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['image_name', 'IoU']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for gt_item in ground_truth:
            gt_key = _get_gt_key(gt_item)
            gt_norm = _norm_id(gt_key)
            gt_bboxes = _get_gt_bboxes(gt_item)
            gt_h, gt_w = _get_gt_size(gt_item)
            orig_h = gt_h or original_size
            gt_processed_bboxes = [preprocess_bbox(bbox, orig_h, image_size)
                                   for bbox in gt_bboxes]

            pred_bboxes = pred_by_key.get(gt_norm, [])
            if not pred_bboxes:
                continue

            try:
                for gt_bbox in gt_processed_bboxes:
                    for pred_bbox in pred_bboxes:
                        iou = computeIoU(gt_bbox, pred_bbox)
                        iou_list.append(iou)
                        writer.writerow({'image_name': gt_key, 'IoU': iou})
            except Exception as e:
                print(f"[average_iou] error for {gt_key}: {e}")
                print(f"  gt_bbox: {gt_processed_bboxes}, pred_bboxes: {pred_bboxes}")

    average_iou_val = sum(iou_list) / len(iou_list) if iou_list else 0
    print(f"Average IoU for dataset {dataset_name}: {average_iou_val:.4f}")
    return average_iou_val


#################################
##############Dice ##############
#################################

def _bin_mask_from_path(path, target_size=None):
    """Load a binary mask file and return a 0/1 numpy array of dtype uint8.

    If ``target_size = (H, W)`` is given, the mask is resized via nearest-neighbor
    so that prediction and ground-truth align before Dice is computed.
    """
    import numpy as np
    from PIL import Image as _Image

    pil = _Image.open(path).convert("L")
    if target_size is not None:
        pil = pil.resize((target_size[1], target_size[0]),
                         resample=_Image.NEAREST)
    arr = np.array(pil, dtype=np.uint8)
    return (arr > 127).astype(np.uint8)


def _dice_score(pred_bin, gt_bin) -> float:
    """Compute Dice between two equal-shape 0/1 numpy arrays."""
    p = pred_bin.reshape(-1).astype("float32")
    g = gt_bin.reshape(-1).astype("float32")
    inter = (p * g).sum()
    denom = p.sum() + g.sum()
    if denom < 1e-6:
        return 1.0  # both empty → perfect agreement
    return float(2.0 * inter / denom)


def average_dice(gt_pth, pred_pth, dataset_name, csv_filename):
    """Compute average Dice between predicted masks and per-frame instance masks.

    Expected JSON layouts
    ---------------------
    Ground truth (unified schema, one record per frame):
        {"image_id": "...", "tasks": {"masks": [{"mask_path": "..."} , ...]}}

    Prediction (one record per frame):
        {"image_id": "...", "mask_paths": ["...", "..."]}

    For frames with multiple instances, predicted masks are merged to a union,
    and ground-truth masks likewise, before Dice is computed.  This keeps the
    metric simple and robust to mask-instance ordering mismatches.
    """
    import numpy as np

    with open(gt_pth, 'r') as file:
        ground_truth = json.load(file)
    with open(pred_pth, 'r') as file:
        predictions = json.load(file)

    # Index predictions by image_id for O(1) lookup
    if isinstance(predictions, dict):
        pred_index = {k: v for k, v in predictions.items()}
    else:
        pred_index = {item["image_id"]: item for item in predictions}

    dice_scores = []
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['image_id', 'Dice']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for gt_item in ground_truth:
            image_id = gt_item.get("image_id")
            gt_masks_meta = (gt_item.get("tasks", {}) or {}).get("masks", []) or []
            gt_paths = [m.get("mask_path") for m in gt_masks_meta if m.get("mask_path")]
            if not gt_paths:
                continue

            pred_entry = pred_index.get(image_id)
            if pred_entry is None:
                # No prediction for this frame — count as Dice 0.
                writer.writerow({'image_id': image_id, 'Dice': 0.0})
                dice_scores.append(0.0)
                continue
            pred_paths = (pred_entry["mask_paths"] if isinstance(pred_entry, dict)
                          else pred_entry)
            if not pred_paths:
                writer.writerow({'image_id': image_id, 'Dice': 0.0})
                dice_scores.append(0.0)
                continue

            try:
                gt_union = _bin_mask_from_path(gt_paths[0])
                target_shape = gt_union.shape
                for p in gt_paths[1:]:
                    gt_union = np.maximum(gt_union, _bin_mask_from_path(p))
                pred_union = _bin_mask_from_path(pred_paths[0],
                                                 target_size=target_shape)
                for p in pred_paths[1:]:
                    pred_union = np.maximum(
                        pred_union,
                        _bin_mask_from_path(p, target_size=target_shape))
                d = _dice_score(pred_union, gt_union)
            except (FileNotFoundError, OSError):
                d = 0.0

            dice_scores.append(d)
            writer.writerow({'image_id': image_id, 'Dice': d})

    avg = sum(dice_scores) / len(dice_scores) if dice_scores else 0.0
    print(f"Average Dice for dataset {dataset_name}: {avg:.4f}")
    return avg


#################################
########### Identify ############
#################################

def identify_accuracy(gt_pth, pred_pth, dataset_name, csv_filename):
    """Compute exact-match accuracy on the [identify] task.

    Ground truth : list of {"image_id": "...", "answer": "lesion"}  (or "nodule")
    Prediction   : dict {image_id: [model_answer_str, ...]} or list of similar.
    """
    with open(gt_pth, 'r') as f:
        gts = json.load(f)
    with open(pred_pth, 'r') as f:
        preds = json.load(f)

    if isinstance(preds, dict):
        pred_index = {k: (v[0] if isinstance(v, list) else v) for k, v in preds.items()}
    else:
        pred_index = {item["image_id"]: item.get("answer", "") for item in preds}

    rows = []
    n_correct = 0
    n_total = 0
    for gt in gts:
        img_id = gt["image_id"]
        gt_label = str(gt.get("answer", "")).strip().lower()
        if not gt_label:
            continue
        pred = str(pred_index.get(img_id, "")).strip().lower()
        # Match if the gold label appears anywhere in the prediction.
        is_correct = (gt_label in pred) if pred else False
        n_total += 1
        n_correct += int(is_correct)
        rows.append({"image_id": img_id, "gt": gt_label,
                     "pred": pred, "correct": int(is_correct)})

    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile,
                                fieldnames=["image_id", "gt", "pred", "correct"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    acc = n_correct / n_total if n_total > 0 else 0.0
    print(f"Identify Accuracy for dataset {dataset_name}: "
          f"{acc:.4f}  ({n_correct}/{n_total})")
    return acc


# ============================================================================ #
#                          Clinical NLG metrics                                 #
# ---------------------------------------------------------------------------- #
# The following metrics extend report-generation evaluation beyond BERT-Sim:
#
#   1. CheXbert-F1  — 14-class chest X-ray disease label agreement
#   2. BLEU-4        — standard n-gram precision
#   3. ROUGE-L       — longest common subsequence recall
#
# They are computed on (prediction, ground-truth) text pairs and exported as
# CSV for per-sample inspection.
# ============================================================================ #


# ---------------------------------------------------------------------------- #
#  CheXbert-style 14-label extraction + F1
# ---------------------------------------------------------------------------- #
# The extractor lives in ``eval_scripts.chexbert`` (dependency-light) so that
# API-side scripts can compute CheXbert-F1 without importing this module's
# heavy sentence_transformers / minigpt4 registration chain.
from eval_scripts.chexbert import (
    CHEXBERT_LABELS,
    extract_chexbert_labels,
    chexbert_f1,
)
# ---------------------------------------------------------------------------- #
#  BLEU-4
# ---------------------------------------------------------------------------- #
import math as _math
from collections import Counter as _Counter


def _ngram_counts(tokens: list, n: int) -> dict:
    return _Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _modified_precision(pred_tokens: list, gt_tokens: list, n: int) -> float:
    pred_counts = _ngram_counts(pred_tokens, n)
    gt_counts = _ngram_counts(gt_tokens, n)
    if not pred_counts:
        return 0.0
    clipped = sum(min(c, gt_counts.get(ng, 0)) for ng, c in pred_counts.items())
    total = sum(pred_counts.values())
    return clipped / total if total > 0 else 0.0


def _brevity_penalty(pred_len: int, gt_len: int) -> float:
    if pred_len == 0:
        return 0.0
    if pred_len > gt_len:
        return 1.0
    return _math.exp(1 - gt_len / pred_len)


def bleu4(pred_text: str, gt_text: str) -> float:
    """Compute BLEU-4 score for a single (prediction, ground-truth) pair."""
    pred_tokens = pred_text.lower().split()
    gt_tokens = gt_text.lower().split()
    if not pred_tokens or not gt_tokens:
        return 0.0

    precisions = []
    for n in range(1, 5):
        p = _modified_precision(pred_tokens, gt_tokens, n)
        # Smoothing: add a small epsilon to avoid log(0)
        precisions.append(p if p > 0 else 1e-7)

    geo_mean = _math.exp(sum(_math.log(p) for p in precisions) / 4)
    bp = _brevity_penalty(len(pred_tokens), len(gt_tokens))
    return bp * geo_mean


def average_bleu4(gt_pth: str, pred_pth: str, dataset_name: str,
                  csv_filename: str) -> float:
    """Compute average BLEU-4 over all samples."""
    def _load(path):
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return [(_norm_id(k), " ".join(v) if isinstance(v, list) else str(v))
                    for k, v in data.items()]
        elif isinstance(data, list) and data and isinstance(data[0], dict):
            return [(_norm_id(item.get("image_id", "")),
                     item.get("caption", item.get("answer", "")))
                    for item in data]
        return []

    gt_items = _load(gt_pth)
    pred_items = _load(pred_pth)
    pred_index = {k: v for k, v in pred_items}

    scores = []
    rows = []
    for img_id, gt_text in gt_items:
        pred_text = pred_index.get(img_id, "")
        s = bleu4(pred_text, gt_text)
        scores.append(s)
        rows.append({"image_id": img_id, "BLEU_4": s})

    avg = sum(scores) / len(scores) if scores else 0.0
    pd.DataFrame(rows).to_csv(csv_filename, index=False)
    print(f"BLEU-4 for {dataset_name}: {avg:.4f}")
    return avg


# ---------------------------------------------------------------------------- #
#  ROUGE-L
# ---------------------------------------------------------------------------- #
def _lcs_length(a: list, b: list) -> int:
    """Longest common subsequence length (dynamic programming)."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def rouge_l(pred_text: str, gt_text: str) -> float:
    """Compute ROUGE-L F1 for a single (prediction, ground-truth) pair."""
    pred_tokens = pred_text.lower().split()
    gt_tokens = gt_text.lower().split()
    if not pred_tokens or not gt_tokens:
        return 0.0
    lcs = _lcs_length(pred_tokens, gt_tokens)
    prec = lcs / len(pred_tokens) if pred_tokens else 0.0
    rec = lcs / len(gt_tokens) if gt_tokens else 0.0
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def average_rouge_l(gt_pth: str, pred_pth: str, dataset_name: str,
                    csv_filename: str) -> float:
    """Compute average ROUGE-L F1 over all samples."""
    def _load(path):
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return [(_norm_id(k), " ".join(v) if isinstance(v, list) else str(v))
                    for k, v in data.items()]
        elif isinstance(data, list) and data and isinstance(data[0], dict):
            return [(_norm_id(item.get("image_id", "")),
                     item.get("caption", item.get("answer", "")))
                    for item in data]
        return []

    gt_items = _load(gt_pth)
    pred_items = _load(pred_pth)
    pred_index = {k: v for k, v in pred_items}

    scores = []
    rows = []
    for img_id, gt_text in gt_items:
        pred_text = pred_index.get(img_id, "")
        s = rouge_l(pred_text, gt_text)
        scores.append(s)
        rows.append({"image_id": img_id, "ROUGE_L": s})

    avg = sum(scores) / len(scores) if scores else 0.0
    pd.DataFrame(rows).to_csv(csv_filename, index=False)
    print(f"ROUGE-L for {dataset_name}: {avg:.4f}")
    return avg


# ---------------------------------------------------------------------------- #
#  All-in-one report evaluation
# ---------------------------------------------------------------------------- #
def evaluate_report_generation(gt_pth: str, pred_pth: str,
                               dataset_name: str, output_dir: str) -> dict:
    """Run all report-generation metrics and return a summary dict.

    Metrics computed:
      - BERT-Similarity (sentence embedding cosine)
      - BLEU-4
      - ROUGE-L
      - CheXbert-F1 (micro + macro + per-label)

    Each metric also writes its own per-sample CSV under *output_dir*.
    """
    results = {}

    # 1. BERT-Sim
    bert_csv = os.path.join(output_dir, f"{dataset_name}_bert_sim.csv")
    results["bert_sim"] = report_bert_sim(gt_pth, pred_pth, bert_csv)

    # 2. BLEU-4
    bleu_csv = os.path.join(output_dir, f"{dataset_name}_bleu4.csv")
    results["bleu4"] = average_bleu4(gt_pth, pred_pth, dataset_name, bleu_csv)

    # 3. ROUGE-L
    rouge_csv = os.path.join(output_dir, f"{dataset_name}_rouge_l.csv")
    results["rouge_l"] = average_rouge_l(gt_pth, pred_pth, dataset_name, rouge_csv)

    # 4. CheXbert-F1
    chex_csv = os.path.join(output_dir, f"{dataset_name}_chexbert_f1.csv")
    chex = chexbert_f1(gt_pth, pred_pth, dataset_name, chex_csv)
    results["chexbert_micro_f1"] = chex["micro_f1"]
    results["chexbert_macro_f1"] = chex["macro_f1"]

    print(f"\n{'='*60}")
    print(f"  Report Generation Summary — {dataset_name}")
    print(f"{'='*60}")
    print(f"  BERT-Sim          : {results['bert_sim']:.4f}")
    print(f"  BLEU-4            : {results['bleu4']:.4f}")
    print(f"  ROUGE-L           : {results['rouge_l']:.4f}")
    print(f"  CheXbert micro-F1 : {results['chexbert_micro_f1']:.4f}")
    print(f"  CheXbert macro-F1 : {results['chexbert_macro_f1']:.4f}")
    print(f"{'='*60}\n")

    return results