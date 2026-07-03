"""Converter for inherited MiniGPT-Med datasets (RadVQA / SLAKE / RSNA).

These datasets already have JSON annotations in the original MiniGPT-Med format.
This converter re-emits them in the unified schema, with a few extras:

- RSNA: bbox scale binning is applied (SAR-Loc Innovation I)
- SLAKE: grounded captions are kept as-is

Note: source JSON paths follow the original MiniGPT-Med dataset configs under
`minigpt4/configs/datasets/<name>/<name>.yaml`. The user can override the
`source_ann_path` argument.

The MIMIC-CXR and NLST converters were removed because those datasets are no
longer used in this project (MIMIC dropped, NLST raw images unavailable).
"""

from __future__ import annotations

from pathlib import Path

from ..dataset_utils import (
    bbox_scale, dump_json, load_json,
)


# ============================================================================
# RadVQA — VQA only
# ============================================================================

def convert_radvqa(source_ann_path: str | Path,
                   image_root:      str | Path = "data/radvqa/imgs",
                   output_train:    str | Path = "data/annotations/radvqa_train.json"):
    src = load_json(source_ann_path)
    records = []
    for item in src:
        records.append({
            "image_id":   item["image_name"],
            "image_path": str(Path(image_root) / item["image_name"]),
            "modality":   "mixed",
            "anatomy":    "multi-organ",
            "tasks": {
                "report":           None,
                "vqa": [{"question": item["question"],
                         "answer":   str(item["answer"])}],
                "grounded_caption": None,
                "boxes":            [],
                "masks":            [],
                "K":                None,
            },
            "split": "train",
        })
    dump_json(records, output_train)
    return records


# ============================================================================
# SLAKE — grounded caption + VQA
# ============================================================================

def convert_slake(source_ann_path: str | Path,
                  image_root:      str | Path = "data/slake/imgs",
                  output_train:    str | Path = "data/annotations/slake_train.json"):
    src = load_json(source_ann_path)
    records = []
    for item in src:
        records.append({
            "image_id":   item["folder_name"],
            "image_path": str(Path(image_root) / item["folder_name"]),
            "modality":   "mixed",
            "anatomy":    "multi-organ",
            "tasks": {
                "report":           None,
                "vqa":              [],
                "grounded_caption": item["grounded_caption"],
                "boxes":            [],
                "masks":            [],
                "K":                None,
            },
            "split": "train",
        })
    dump_json(records, output_train)
    return records


# ============================================================================
# RSNA — detection only (eval-only set, but we emit train JSON for completeness)
# ============================================================================

def convert_rsna(source_ann_path: str | Path,
                 image_root:      str | Path = "data/rsna/RSNA-bbox-1024",
                 output_eval:     str | Path = "data/annotations/rsna_eval.json",
                 original_size:   int = 1024):
    """RSNA annotations carry bbox in 1024×1024 pixel coords.

    Per dataset_plan.md §2.5, RSNA is used for zero-shot evaluation only.
    """
    src = load_json(source_ann_path)
    records = []
    for item in src:
        boxes = []
        for bb in item.get("bbox", []):
            x1, y1, x2, y2 = bb
            boxes.append({
                "class":          "pneumonia",
                "bbox":           [int(x1), int(y1), int(x2), int(y2)],
                "scale":          bbox_scale([x1, y1, x2, y2], original_size, original_size),
                "anatomy_region": None,
            })
        records.append({
            "image_id":   item["key"],
            "image_path": str(Path(image_root) / item["key"]),
            "modality":   "CXR",
            "anatomy":    "chest",
            "image_size": [original_size, original_size],
            "tasks": {
                "report":           None,
                "vqa":              [],
                "grounded_caption": None,
                "boxes":            boxes,
                "masks":            [],
                "K":                len(boxes),
            },
            "split": "test",  # ★ eval-only
        })
    dump_json(records, output_eval)
    return records

