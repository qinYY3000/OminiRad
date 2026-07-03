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
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')


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
    
    # Create a list to store BERT similarity data
    bert_similarity_data = []
    
    # Initialize variables to calculate the average
    total_similarity = 0
    total_count = 0
    
    # Iterate over each item in the prediction_data list
    for item in prediction_data:
        # Extract the image_id and corresponding prediction caption
        image_id = item["image_id"]
        prediction_caption = item["caption"]
        
        # Search for the matching ground truth caption based on image_id
        ground_truth_caption = None
        for gt_item in ground_truth_data:
            if gt_item["image_id"] == image_id:
                ground_truth_caption = gt_item["caption"]
                break
        
        if ground_truth_caption is not None:
            bert_similarity = compute_bert_similarity(prediction_caption, ground_truth_caption)
            bert_similarity_data.append({"image_id": image_id, "BERT_score": bert_similarity})
            
            total_similarity += bert_similarity
            total_count += 1
    
    average_similarity = total_similarity / total_count if total_count > 0 else 0
    
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

    gt_qa_pairs = {(entry['image_name'], entry['question']): entry['answer'] for entry in gt_data}

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
    df = pd.DataFrame(results)
    df_sorted = df.sort_values(by="BERT_score", ascending=True)
    df_sorted.to_csv(output_csv, index=False)
    print(f"Average BERT similarity score: {average_similarity}")


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

    iou_list = []

    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['image_name', 'IoU']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for gt_item in ground_truth:
            gt_key = gt_item['key']
            gt_bboxes = gt_item['bbox']
            original_size = gt_item['height']
            gt_processed_bboxes = [preprocess_bbox(bbox, original_size, image_size) for bbox in gt_bboxes]

            for pred_item in predictions:
                pred_key = pred_item['key'].replace(".png", "")

                if gt_key == pred_key:
                    pred_bboxes = pred_item['bbox']
                    try:
                        for gt_bbox in gt_processed_bboxes:
                            for pred_bbox in pred_bboxes:
                                iou = computeIoU(gt_bbox, pred_bbox)
                                iou_list.append(iou)
                                writer.writerow({'image_name': gt_key, 'IoU': iou})
                                print(gt_key)
                                print(iou)
                    except Exception as e:
                        print("gt_bbox: ", gt_bbox)
                        print("gt_bbox: ", pred_bboxes)

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
# A lightweight regex-based labeler that mimics CheXbert's 14-class output.
# It detects positive / negative mentions of each finding in free-text
# radiology reports and returns a binary vector.  No model weights required.
#
# The 14 CheXpert labels (same order as the original CheXbert):
CHEXBERT_LABELS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

# (regex, label) pairs.  Order matters: more specific patterns first.
# Each regex is case-insensitive.  Negation is handled separately.
_CHEXBERT_PATTERNS = [
    # No Finding — only if no other finding is detected (handled in extractor)
    (r"(enlarged\s+)?cardiomediastin(um|al\s+silhouette)?.{0,30}(enlarg|wide|prominen)",
     "Enlarged Cardiomediastinum"),
    # Cardiomegaly: match "cardiomegaly" or "cardiac silhouette ... enlarged"
    (r"cardiomegaly\b", "Cardiomegaly"),
    (r"cardiac\s+(silhouette|outline).{0,20}(enlarg|prominen)", "Cardiomegaly"),
    (r"lung\s+opacity|airspace\s+opacity|opacit(y|ies)", "Lung Opacity"),
    # Lung Lesion: must be "lung" + (mass/nodule/tumor), not just "nodule" alone
    (r"lung\s+(lesion|mass|nodule|tumor)|pulmonary\s+nodule", "Lung Lesion"),
    (r"edema|pulmonary\s+edema|congestion", "Edema"),
    (r"consolidation", "Consolidation"),
    (r"pneumonia|infiltrate", "Pneumonia"),
    (r"atelectasis|atelectatic", "Atelectasis"),
    (r"pneumothorax", "Pneumothorax"),
    (r"pleural\s+effusion|effusion", "Pleural Effusion"),
    (r"pleural\s+(thickening|plaque|calcification)", "Pleural Other"),
    (r"fracture", "Fracture"),
    (r"(support|enteral|endotracheal)\s+(device|tube|line)|picc|port|pacemaker|defibrillator",
     "Support Devices"),
]

# Negation patterns that flip a positive mention to negative.
_NEGATION_PREFIX = re.compile(
    r"(no|without|absent|no\s+evidence\s+of|none|free\s+of|not\s+seen|"
    r"cannot\s+see|does\s+not\s+show|without\s+evidence\s+of)\s+",
    re.IGNORECASE,
)


def extract_chexbert_labels(text: str) -> list[int]:
    """Return a 14-dim binary vector (1=positive, 0=negative/absent).

    Uses regex pattern matching with negation detection.  This is a
    lightweight approximation of the CheXbert model — it is *not* a
    replacement for the full neural labeler but captures the most common
    clinical findings in chest X-ray reports.
    """
    text_lower = text.lower()
    labels = [0] * len(CHEXBERT_LABELS)

    for pattern, label_name in _CHEXBERT_PATTERNS:
        idx = CHEXBERT_LABELS.index(label_name)
        for m in re.finditer(pattern, text_lower):
            start = max(0, m.start() - 40)
            preceding = text_lower[start:m.start()]
            match_text = m.group(0)
            following = text_lower[m.end():m.end() + 30]

            # Negation before match: "no cardiomegaly", "without consolidation"
            # Also handles "no X or Y" / "no X of Y" patterns by checking the
            # last clause of the preceding text (after sentence boundary).
            last_sentence = re.split(r"[.;]\s+", preceding)[-1] if preceding else ""
            if re.search(
                r"\b(no|without|absent|none|no\s+evidence\s+of|cannot|does\s+not|free\s+of)\b",
                last_sentence,
            ):
                continue  # Negated → treat as absent
            # Negation inside match: "cardiac silhouette is not enlarged"
            if re.search(r"\bnot\s+(enlarg|prominen|present|seen|noted|visualiz)\b",
                         match_text):
                continue
            # Negation after match: "enlarged ... not"
            if re.search(r"\bnot\s+(enlarg|present|seen|noted|visualiz)\b",
                         following):
                continue
            labels[idx] = 1

    # "No Finding" = 1 only when no other label is positive.
    if sum(labels[1:]) == 0:
        if re.search(
            r"\b(normal|no\s+acute|clear\s+lungs|unremarkable|no\s+acute\s+cardiopulmonary)\b",
            text_lower,
        ):
            labels[0] = 1

    return labels


def chexbert_f1(gt_pth: str, pred_pth: str, dataset_name: str,
                csv_filename: str) -> dict:
    """Compute CheXbert-style clinical F1 between prediction and ground-truth.

    Returns a dict with keys: ``micro_f1``, ``macro_f1``, ``per_label_f1``.
    """
    import numpy as np

    def _load(path):
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            # {image_id: [caption]}  (after clean_report_json)
            return [(k, " ".join(v) if isinstance(v, list) else str(v))
                    for k, v in data.items()]
        elif isinstance(data, list) and data and isinstance(data[0], dict):
            return [(item.get("image_id", ""), item.get("caption", item.get("answer", "")))
                    for item in data]
        return []

    gt_items = _load(gt_pth)
    pred_items = _load(pred_pth)
    pred_index = {k: v for k, v in pred_items}

    all_gt_labels = []
    all_pred_labels = []

    rows = []
    for img_id, gt_text in gt_items:
        pred_text = pred_index.get(img_id, "")
        gt_labels = extract_chexbert_labels(gt_text)
        pred_labels = extract_chexbert_labels(pred_text)
        all_gt_labels.append(gt_labels)
        all_pred_labels.append(pred_labels)
        rows.append({"image_id": img_id, **{
            f"gt_{l}": gt_labels[i] for i, l in enumerate(CHEXBERT_LABELS)
        }, **{
            f"pred_{l}": pred_labels[i] for i, l in enumerate(CHEXBERT_LABELS)
        }})

    if not all_gt_labels:
        print(f"[chexbert_f1] no samples for {dataset_name}")
        return {"micro_f1": 0.0, "macro_f1": 0.0, "per_label_f1": {}}

    gt_arr = np.array(all_gt_labels)      # (N, 14)
    pred_arr = np.array(all_pred_labels)  # (N, 14)

    # Per-label F1
    per_label_f1 = {}
    f1_sum = 0.0
    n_active = 0
    tp_total = fp_total = fn_total = 0

    for i, label in enumerate(CHEXBERT_LABELS):
        gt_col = gt_arr[:, i]
        pred_col = pred_arr[:, i]
        tp = int(((gt_col == 1) & (pred_col == 1)).sum())
        fp = int(((gt_col == 0) & (pred_col == 1)).sum())
        fn = int(((gt_col == 1) & (pred_col == 0)).sum())
        tp_total += tp; fp_total += fp; fn_total += fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        per_label_f1[label] = round(f1, 4)

        # Macro-F1 only counts labels that appear in GT
        if gt_col.sum() > 0:
            f1_sum += f1
            n_active += 1

    # Micro-F1
    micro_prec = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    micro_rec = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    micro_f1 = (2 * micro_prec * micro_rec / (micro_prec + micro_rec)
                if (micro_prec + micro_rec) > 0 else 0.0)
    macro_f1 = f1_sum / n_active if n_active > 0 else 0.0

    # Write CSV
    df = pd.DataFrame(rows)
    df.to_csv(csv_filename, index=False)

    print(f"CheXbert-F1 for {dataset_name}: "
          f"micro_f1={micro_f1:.4f}  macro_f1={macro_f1:.4f}")
    for label, f1 in per_label_f1.items():
        if f1 > 0 or "No Finding" not in label:
            print(f"  {label:<30s} F1={f1:.4f}")

    return {"micro_f1": micro_f1, "macro_f1": macro_f1,
            "per_label_f1": per_label_f1}


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
            return [(k, " ".join(v) if isinstance(v, list) else str(v))
                    for k, v in data.items()]
        elif isinstance(data, list) and data and isinstance(data[0], dict):
            return [(item.get("image_id", ""), item.get("caption", item.get("answer", "")))
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
            return [(k, " ".join(v) if isinstance(v, list) else str(v))
                    for k, v in data.items()]
        elif isinstance(data, list) and data and isinstance(data[0], dict):
            return [(item.get("image_id", ""), item.get("caption", item.get("answer", "")))
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