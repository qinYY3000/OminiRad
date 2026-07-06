"""Common utilities shared by all dataset converters.

Includes:
- mask → bbox derivation (single + multi-instance)
- bbox scale binning (S/M/L) — see innovation.md §3.4.1
- K=0 healthy report mining — see dataset_plan.md §5.4
- VQA derivation from masks/reports — see dataset_plan.md §5.5
- Anatomy NER (thin wrapper around data.anatomy_lexicon)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
# `skimage` is heavy and only needed for split_instances(); import lazily there.


# ============================================================================
# Mask → bbox
# ============================================================================

def mask_to_bbox(mask_path: str | Path, min_area: int = 50) -> list[int] | None:
    """Return tight bounding box [x1, y1, x2, y2] (pixel coords) of a binary mask.

    Returns None if the mask is empty or all components are too small.
    """
    m = np.asarray(Image.open(mask_path).convert("L")) > 127
    if m.sum() < min_area:
        return None
    ys, xs = np.where(m)
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def split_instances(mask_path: str | Path,
                    output_dir: str | Path,
                    image_id: str,
                    min_area: int = 50) -> list[Path]:
    """Split a merged binary mask into per-instance masks via connected components.

    Writes individual mask PNGs named `<image_id>_inst{i}.png` under output_dir.
    Returns the list of written mask paths.
    """
    from skimage import measure  # lazy import — heavy dependency
    m = np.asarray(Image.open(mask_path).convert("L")) > 127
    labels = measure.label(m, connectivity=2)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for inst_id in range(1, labels.max() + 1):
        inst_mask = (labels == inst_id)
        if inst_mask.sum() < min_area:
            continue
        out_path = output_dir / f"{image_id}_inst{inst_id - 1}.png"
        Image.fromarray((inst_mask * 255).astype(np.uint8)).save(out_path)
        written.append(out_path)
    return written


def masks_to_boxes(mask_paths: Iterable[str | Path],
                   image_w: int,
                   image_h: int,
                   class_name: str = "lesion") -> list[dict]:
    """Convert a list of per-instance binary mask files → list of bbox dicts.

    Each output dict has: class, bbox, scale (auto from area).
    Masks producing empty/too-small bboxes are skipped.
    """
    boxes = []
    for p in mask_paths:
        bbox = mask_to_bbox(p)
        if bbox is None:
            continue
        boxes.append({
            "class": class_name,
            "bbox":  bbox,
            "scale": bbox_scale(bbox, image_w, image_h),
        })
    return boxes


# ============================================================================
# BBox scale binning (SAR-Loc Innovation I)
# ============================================================================

def bbox_scale(bbox: list[int], image_w: int, image_h: int) -> str:
    """Bin a bbox into Small / Medium / Large by area ratio.

    See innovation.md §3.4.1 — these thresholds are chosen to balance the three
    SAR-Loc heads when training across radiology (small nodules) and US
    (medium lesions) modalities.
    """
    x1, y1, x2, y2 = bbox
    area_ratio = ((x2 - x1) * (y2 - y1)) / float(image_w * image_h + 1e-9)
    if   area_ratio < 0.01: return "S"   # < 1% — small nodule, micro-calcification
    elif area_ratio < 0.10: return "M"   # 1-10% — typical US lesions, consolidations
    else:                   return "L"   # >10% — cardiomegaly, pleural effusion


# ============================================================================
# K=0 healthy report mining (MIMIC-CXR only, per dataset_plan.md §9.3)
# ============================================================================

_HEALTHY_PHRASES = [
    "no acute findings", "no acute cardiopulmonary",
    "unremarkable", "no significant abnormality",
    "within normal limits", "no evidence of",
    "no focal consolidation", "no pleural effusion",
    "no pneumothorax", "clear lungs",
]

_PATHOLOGY_TERMS = [
    "consolidation", "effusion", "pneumothorax", "nodule",
    "mass", "opacity", "fracture", "edema", "infiltrate",
    "atelectasis", "cardiomegaly", "lesion",
]


def is_healthy_report(text: str) -> bool:
    """Return True if the report appears to describe a healthy CXR.

    Heuristic: at least one HEALTHY phrase AND no PATHOLOGY term.
    """
    if not text:
        return False
    t = text.lower()
    has_healthy   = any(p in t for p in _HEALTHY_PHRASES)
    has_pathology = any(b in t for b in _PATHOLOGY_TERMS)
    return has_healthy and not has_pathology


# ============================================================================
# VQA derivation from masks + reports (deterministic, no LLM hallucination)
# ============================================================================

def derive_vqa(report: str | None,
               boxes: list[dict],
               anatomy: str) -> list[dict]:
    """Generate a small set of template VQA pairs from structured info.

    See dataset_plan.md §5.5. Yields ~2-4 QAs per sample.
    """
    qa: list[dict] = []
    K = len(boxes)

    # Cardinality question
    if K == 0:
        qa.append({"question": "How many lesions are visible in this image?",
                   "answer": "none"})
    else:
        qa.append({"question": "How many lesions are visible in this image?",
                   "answer": str(K)})

    # Location question (uses anatomy_region populated by anatomy NER)
    if K >= 1:
        region = boxes[0].get("anatomy_region")
        if region:
            qa.append({"question": "Where is the lesion located?",
                       "answer": region.replace("_", " ")})

    # Modality / anatomy sanity question (cheap but useful for VQA training mix)
    if anatomy == "breast":
        qa.append({"question": "What imaging modality is this?",
                   "answer": "breast ultrasound"})
    elif anatomy == "thyroid":
        qa.append({"question": "What imaging modality is this?",
                   "answer": "thyroid ultrasound"})
    elif anatomy == "colon":
        qa.append({"question": "What imaging modality is this?",
                   "answer": "colonoscopy"})

    # Side question (left/right) — derivable from anatomy_region
    if K >= 1:
        region = boxes[0].get("anatomy_region") or ""
        if region.startswith("right_"):
            qa.append({"question": "Which side is the lesion on?",
                       "answer": "right"})
        elif region.startswith("left_"):
            qa.append({"question": "Which side is the lesion on?",
                       "answer": "left"})

    return qa


# ============================================================================
# JSON I/O
# ============================================================================

def dump_json(records: list[dict], path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"[dump_json] wrote {len(records)} records → {path}")


def load_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================================
# Patient-level split (per dataset_plan.md §2.6 / §2.7)
# ============================================================================

def patient_level_split(records: list[dict],
                        ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
                        seed: int = 42) -> tuple[list[dict], list[dict], list[dict]]:
    """Split records into train/val/test grouped by `patient_id`.

    All records of the same patient stay in the same split.
    """
    import random

    by_pid: dict[str, list[dict]] = {}
    for r in records:
        pid = r.get("patient_id") or r["image_id"]   # fallback to image_id
        by_pid.setdefault(pid, []).append(r)

    pids = sorted(by_pid.keys())
    rng = random.Random(seed)
    rng.shuffle(pids)

    n_total = len(pids)
    n_train = int(round(n_total * ratios[0]))
    n_val   = int(round(n_total * ratios[1]))
    train_pids = set(pids[:n_train])
    val_pids   = set(pids[n_train:n_train + n_val])

    train, val, test = [], [], []
    for pid, recs in by_pid.items():
        if   pid in train_pids: bucket = train
        elif pid in val_pids:   bucket = val
        else:                   bucket = test
        for r in recs:
            r["split"] = ("train" if bucket is train else
                          "val"   if bucket is val   else "test")
            bucket.append(r)

    print(f"[split] patients: train={len(train_pids)} val={len(val_pids)} "
          f"test={n_total - len(train_pids) - len(val_pids)} | "
          f"frames: train={len(train)} val={len(val)} test={len(test)}")
    return train, val, test


__all__ = [
    "mask_to_bbox", "split_instances", "masks_to_boxes", "bbox_scale",
    "is_healthy_report", "derive_vqa",
    "dump_json", "load_json", "patient_level_split",
]
