"""Converter for the Kvasir-SEG polyp dataset (endoscopy).

Produces unified-schema JSON annotations for 5 tasks:
  segmentation / detection / refer / identify / vqa

Kvasir-SEG images and masks are one-to-one by ``image_id``:
  - Images: ``data/kvasir/imgs/<image_id>.jpg``
  - Masks:  ``data/kvasir/masks/<image_id>.jpg``

Mask paths are written into the JSON immediately — no mask files need to
exist at conversion time.  The dataset class lazily loads masks at training
time; missing masks are silently skipped.

No mask splitting or instance caching needed: Kvasir-SEG masks are already
per-instance (one polyp per bbox, though a single image may have multiple
polyps — each represented as a separate bbox entry in kavsir_bboxes.json).
Since the dataset provides per-polyp bboxes but only a single merged mask
per image, we keep the merged mask path for segmentation and derive bboxes
from the raw JSON.

VQA pairs follow the question style of Kvasir metadata.csv.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..dataset_utils import bbox_scale, dump_json, patient_level_split

ANATOMY = "colon"
CLASS_LABEL = "polyp"
MODALITY = "endoscopy"

# Mask filename is always <image_id>.jpg, same as the image.
# Kvasir-SEG images and masks share the same filename convention.
MASK_DIR = "data/kvasir/masks"


# ---------------------------------------------------------------------------
# Position / size helpers for VQA answers
# ---------------------------------------------------------------------------

def _bbox_relative_position(bbox: list[int], w: int, h: int) -> str:
    """Map bbox center to a clinical position string (like metadata.csv)."""
    cx = (bbox[0] + bbox[2]) / 2 / max(w, 1)
    cy = (bbox[1] + bbox[3]) / 2 / max(h, 1)
    vy = "upper" if cy < 0.33 else ("lower" if cy > 0.67 else "center")
    vx = "left" if cx < 0.33 else ("right" if cx > 0.67 else "center")
    if vy == "center" and vx == "center":
        return "center"
    if vy == "center":
        return vx
    if vx == "center":
        return f"{vy}-center"
    return f"{vy}-{vx}"


def _polyp_size_from_scale(scale: str) -> str:
    return {"S": "small", "M": "medium", "L": "large"}.get(scale, "medium")


# ---------------------------------------------------------------------------
# VQA generation (style aligned with Kvasir metadata.csv)
# ---------------------------------------------------------------------------

def _generate_vqa(boxes: list[dict], w: int, h: int) -> list[dict]:
    K = len(boxes)
    qa: list[dict] = []

    qa.append({"question": "How many polyps are in the image?", "answer": str(K)})
    qa.append({"question": "How many findings are present?", "answer": str(K)})

    if K > 0:
        qa.append({"question": "Are there any abnormalities in the image? Check all that are present.", "answer": "polyp"})
        qa.append({"question": "Does this image contain any finding?", "answer": "yes"})
        qa.append({"question": "Have all polyps been removed?", "answer": "no"})
    else:
        qa.append({"question": "Are there any abnormalities in the image? Check all that are present.", "answer": "none"})
        qa.append({"question": "Does this image contain any finding?", "answer": "no"})
        qa.append({"question": "Have all polyps been removed?", "answer": "yes"})

    qa.append({"question": "What type of polyp is present?", "answer": "none" if K == 0 else CLASS_LABEL})
    qa.append({"question": "What type of procedure is the image taken from?", "answer": "colonoscopy"})

    if K >= 1:
        scales = [b.get("scale", "M") for b in boxes]
        has_easy = any(s == "L" for s in scales)
        has_hard = any(s == "S" for s in scales)
        detectability = "some are easy, some are difficult" if (has_easy and has_hard) else ("yes" if has_easy else "no")
        qa.append({"question": "Is this finding easy to detect?", "answer": detectability})

        if K == 1:
            qa.append({"question": "What is the size of the polyp?", "answer": _polyp_size_from_scale(scales[0])})
        else:
            sizes_str = "; ".join(f"polyp {i+1}: {_polyp_size_from_scale(scales[i])}" for i in range(min(K, len(scales))))
            qa.append({"question": "What is the size of the polyp?", "answer": sizes_str})

        positions = [_bbox_relative_position(b["bbox"], w, h) for b in boxes]
        qa.append({"question": "Where in the image is the abnormality?", "answer": "; ".join(positions)})

    qa.append({"question": "How many instruments are in the image?", "answer": "0"})
    qa.append({"question": "Are there any instruments in the image? Check all that are present.", "answer": "none"})
    qa.append({"question": "How many instrumnets are in the image?", "answer": "0"})

    return qa


# ---------------------------------------------------------------------------
# Main converter
# ---------------------------------------------------------------------------

def _iter_samples(raw_bbox_path: str | Path = "data/kvasir/kavsir_bboxes.json") -> list[dict]:
    raw_bbox_path = Path(raw_bbox_path)
    with open(raw_bbox_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    samples: list[dict] = []
    for image_id, entry in sorted(raw.items()):
        bboxes_raw = entry.get("bbox", [])
        if not bboxes_raw:
            continue

        h = entry.get("height", 0)
        w = entry.get("width", 0)

        # Bboxes from raw JSON
        boxes = []
        for b in bboxes_raw:
            x1, y1, x2, y2 = int(b["xmin"]), int(b["ymin"]), int(b["xmax"]), int(b["ymax"])
            scale = bbox_scale([x1, y1, x2, y2], w, h)
            boxes.append({
                "class": b.get("label", CLASS_LABEL),
                "bbox": [x1, y1, x2, y2],
                "scale": scale,
                "anatomy_region": ANATOMY,
            })

        K = len(boxes)

        # Mask path: always written, loaded lazily by dataset class.
        # Kvasir-SEG masks and images share the same <image_id>.jpg filename.
        masks = [{"class": CLASS_LABEL, "mask_path": f"{MASK_DIR}/{image_id}.jpg"}]

        vqa = _generate_vqa(boxes, w, h)

        samples.append({
            "image_id": image_id,
            "image_path": f"data/kvasir/imgs/{image_id}.jpg",
            "modality": MODALITY,
            "anatomy": ANATOMY,
            "image_size": [w, h],
            "tasks": {
                "report": None,
                "vqa": vqa,
                "boxes": boxes,
                "masks": masks,
                "K": K,
            },
            "patient_id": image_id,
        })

    print(f"[kvasir] collected {len(samples)} frames")
    return samples


def convert(
    raw_bbox_path: str | Path = "data/kvasir/kavsir_bboxes.json",
    output_train: str | Path = "data/annotations/kvasir_train.json",
    output_val: str | Path = "data/annotations/kvasir_val.json",
    output_test: str | Path = "data/annotations/kvasir_test.json",
    seed: int = 42,
) -> list[dict]:
    """Convert Kvasir-SEG → unified-schema train/val/test JSONs.

    Mask paths are always written; the dataset class handles missing masks
    gracefully (skips segmentation loss for those samples).
    """
    samples = _iter_samples(raw_bbox_path=raw_bbox_path)
    train, val, test = patient_level_split(samples, ratios=(0.76, 0.04, 0.20), seed=seed)
    dump_json(train, output_train)
    dump_json(val, output_val)
    dump_json(test, output_test)
    return samples
