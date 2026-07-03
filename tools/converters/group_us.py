"""Converter for the private Group-Ultrasound datasets (breast & thyroid).

This converter is built to handle two assumed source layouts (you can adjust
`SCAN_PATTERN` if your raw data is organized differently):

    data/group_breast/
        frames/<patient_id>/<study_id>/<frame_idx>.png
        masks/<patient_id>/<study_id>/<frame_idx>.png         # merged mask
        reports/<patient_id>/<study_id>.txt                   # plain text report

    data/group_thyroid/ (same layout)

For every frame, we:
  1. Connected-component split the merged mask into per-instance mask files,
     written to  `masks_instances/<patient_id>/<study_id>/<frame_idx>_inst{k}.png`
  2. Derive bbox + scale from each instance mask
  3. Run anatomy NER on the report and tag each box with `anatomy_region`
  4. Derive a small set of template VQA pairs
  5. Emit one unified-schema JSON record per frame

Per dataset_plan.md §9.1: every frame of a study is paired with the same report.
Per dataset_plan.md §9.3: no K=0 here (private US has no normal scans).
"""

from __future__ import annotations

from pathlib import Path

from ..dataset_utils import (
    split_instances, masks_to_boxes, derive_vqa,
    dump_json, patient_level_split,
)

# Anatomy NER (imported lazily so this file works even if data/ isn't on sys.path)
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.anatomy_lexicon import extract_anatomy_auto  # noqa: E402


# Per-instance mask cache directory (will be written if not present)
INSTANCE_SUBDIR = "masks_instances"


def _read_report(report_path: Path) -> str | None:
    if not report_path.exists():
        return None
    return report_path.read_text(encoding="utf-8").strip()


def _iter_frames(root: Path):
    """Yield (patient_id, study_id, frame_path) for every frame under root/frames/."""
    frames_root = root / "frames"
    if not frames_root.exists():
        raise FileNotFoundError(f"frames/ not found under {root}")
    for pid_dir in sorted(p for p in frames_root.iterdir() if p.is_dir()):
        for study_dir in sorted(s for s in pid_dir.iterdir() if s.is_dir()):
            for frame_path in sorted(study_dir.glob("*.png")):
                yield pid_dir.name, study_dir.name, frame_path


def _grounded_caption_from_boxes(boxes: list[dict],
                                  class_label: str) -> str | None:
    """Build a simple SLAKE-style inline-bbox caption from the bbox list."""
    if not boxes:
        return None
    parts = []
    for b in boxes:
        x1, y1, x2, y2 = b["bbox"]
        # SLAKE uses normalized 0-100 coords inside braces
        # We keep pixel coords here and let the dataset loader normalize.
        parts.append(f"<p>{class_label}</p> {{<{x1}><{y1}><{x2}><{y2}>}}")
    return " ".join(parts)


def convert(root: str | Path,
            anatomy: str,
            class_label: str,
            output_train: str | Path,
            output_val: str | Path,
            output_test: str | Path,
            target_size: int = 448,
            seed: int = 42) -> list[dict]:
    """Convert the private US dataset rooted at `root` into the unified schema.

    Writes three JSON files (train / val / test). Patient-level splitting at
    76 / 4 / 20 — the val split enables best-checkpoint selection.

    Parameters
    ----------
    root         : path to data/group_breast or data/group_thyroid
    anatomy      : "breast" or "thyroid"
    class_label  : "lesion" or "nodule"
    """
    from PIL import Image

    root = Path(root)
    instance_root = root / INSTANCE_SUBDIR
    instance_root.mkdir(exist_ok=True)

    records: list[dict] = []
    skipped_no_mask  = 0
    skipped_empty    = 0

    for pid, study, frame_path in _iter_frames(root):
        # Find merged mask
        mask_path = root / "masks" / pid / study / frame_path.name
        if not mask_path.exists():
            skipped_no_mask += 1
            continue

        # Get image size
        with Image.open(frame_path) as im:
            image_w, image_h = im.size

        # 1) Split merged mask into instance masks (cached on disk)
        instance_dir = instance_root / pid / study
        image_id = f"group_{anatomy}_{pid}_{study}_{frame_path.stem}"
        inst_paths = split_instances(mask_path, instance_dir, image_id)
        if not inst_paths:
            skipped_empty += 1
            continue

        # 2) Derive bboxes (with scale) from instance masks
        boxes = masks_to_boxes(inst_paths, image_w, image_h, class_name=class_label)
        if not boxes:
            skipped_empty += 1
            continue

        # 3) Load shared report (one per study) and run anatomy NER
        report_path = root / "reports" / pid / f"{study}.txt"
        report = _read_report(report_path)
        anatomy_region = extract_anatomy_auto(report or "", anatomy)
        for b in boxes:
            b["anatomy_region"] = anatomy_region

        # 4) Build mask refs (keep paths)
        masks = [{"class": class_label, "mask_path": str(p)} for p in inst_paths]

        # 5) Derive template VQA pairs
        vqa = derive_vqa(report, boxes, anatomy)

        # 6) Grounded caption (optional, only if anatomy_region was found)
        grounded = _grounded_caption_from_boxes(boxes, class_label)

        records.append({
            "image_id":   image_id,
            "image_path": str(frame_path),
            "modality":   "ultrasound",
            "anatomy":    anatomy,
            "image_size": [image_w, image_h],
            "tasks": {
                "report":           report,
                "vqa":              vqa,
                "grounded_caption": grounded,
                "boxes":            boxes,
                "masks":            masks,
                "K":                len(boxes),
            },
            "patient_id": pid,
            "study_id":   study,
            # split assigned below
        })

    print(f"[group_us:{anatomy}] collected {len(records)} frames  "
          f"(skipped {skipped_no_mask} no-mask, {skipped_empty} empty)")

    # 7) Three-way patient-level split: 76% train / 4% val / 20% test.
    train, val, test = patient_level_split(
        records, ratios=(0.76, 0.04, 0.20), seed=seed
    )
    dump_json(train, output_train)
    dump_json(val,   output_val)
    dump_json(test,  output_test)
    return records


def convert_breast(root="data/group_breast"):
    return convert(
        root=root, anatomy="breast", class_label="lesion",
        output_train="data/annotations/group_breast_train.json",
        output_val  ="data/annotations/group_breast_val.json",
        output_test ="data/annotations/group_breast_test.json",
    )


def convert_thyroid(root="data/group_thyroid"):
    return convert(
        root=root, anatomy="thyroid", class_label="nodule",
        output_train="data/annotations/group_thyroid_train.json",
        output_val  ="data/annotations/group_thyroid_val.json",
        output_test ="data/annotations/group_thyroid_test.json",
    )
