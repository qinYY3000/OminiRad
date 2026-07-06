"""Converter for the Indiana University Chest X-Ray (Open-i) dataset.

Source layout (after extracting the version-2 archive)::

    data/Chest X/
        indiana_projections.csv      # uid, filename, projection
        indiana_reports.csv          # uid, MeSH, Problems, image,
                                     # indication, comparison, findings, impression
        images/                      # contains all *.dcm.png frames

Behavior
--------
* Each record in ``indiana_projections.csv`` becomes **one** unified-schema
  record (one image == one sample).
* The shared report (``findings`` + ``impression`` from ``indiana_reports.csv``)
  is broadcast to every image of the same study (``uid``).
* Open-i de-identifies named entities with the literal token ``XXXX``.  We
  collapse repeated ``XXXX`` runs and replace the single ``XXXX`` token with
  ``[REDACTED]`` so that captions stay readable for both training and BERT-Sim
  evaluation.
* Patient-level split: 76 % train / 4 % val / 20 % test, grouped by ``uid``
  so that the two views (Frontal + Lateral) of the same patient stay in the
  same split.  The val split enables best-checkpoint selection.
* The emitted records expose ``image_id`` and ``caption`` fields consumed by
  :class:`IndianaCXRDataset` (file extension ``.png`` is appended at load
  time).
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

from ..dataset_utils import dump_json, patient_level_split


# Open-i replaces all PHI tokens with the literal string "XXXX".
# We collapse repeats so reports read naturally; we keep a single placeholder
# so that BERT-Sim does not over-reward empty strings.
_XXXX_RUN = re.compile(r"(?:\bXXXX\b[\s,.;:-]*){2,}")
_MULTI_SPACE = re.compile(r"\s{2,}")


def _clean(text: str | None) -> str:
    if not text:
        return ""
    cleaned = _XXXX_RUN.sub("XXXX ", str(text))
    cleaned = cleaned.replace("XXXX", "[REDACTED]")
    cleaned = _MULTI_SPACE.sub(" ", cleaned).strip()
    return cleaned


def _build_caption(findings: str, impression: str) -> str:
    """Combine findings + impression into a single free-form caption."""
    findings = _clean(findings)
    impression = _clean(impression)
    if findings and impression:
        return f"{findings} Impression: {impression}"
    return findings or impression


def _read_reports_csv(path: Path) -> dict[str, dict]:
    """uid → row dict (cleaned)."""
    out: dict[str, dict] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = (row.get("uid") or "").strip()
            if not uid:
                continue
            out[uid] = {
                "mesh":       row.get("MeSH", "") or "",
                "problems":   row.get("Problems", "") or "",
                "indication": row.get("indication", "") or "",
                "comparison": row.get("comparison", "") or "",
                "findings":   row.get("findings", "") or "",
                "impression": row.get("impression", "") or "",
            }
    return out


def _read_projections_csv(path: Path) -> list[dict]:
    """uid + filename + projection rows, in source order."""
    out: list[dict] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = (row.get("uid") or "").strip()
            fname = (row.get("filename") or "").strip()
            proj = (row.get("projection") or "").strip()
            if not uid or not fname:
                continue
            out.append({"uid": uid, "filename": fname, "projection": proj})
    return out


def convert(root: str | Path = "data/Chest X",
            image_subdir: str = "images",
            output_train: str | Path = "data/annotations/indiana_train.json",
            output_val:   str | Path = "data/annotations/indiana_val.json",
            output_test:  str | Path = "data/annotations/indiana_test.json",
            seed: int = 42,
            min_caption_chars: int = 5) -> list[dict]:
    """Convert Indiana Open-i CSV pair into unified-schema JSON files.

    Three-way patient-level split: 76 % train / 4 % val / 20 % test.
    The val split (5 % of what was the train portion) enables best-checkpoint
    selection during training.

    Parameters
    ----------
    root : path containing both CSVs and the ``images/`` folder.
    image_subdir : directory name (relative to ``root``) holding ``*.dcm.png``.
    output_train / val / test : paths for the three JSON splits.
    seed : RNG seed for the patient-level split.
    min_caption_chars : skip records whose caption has fewer characters than
        this — these usually indicate empty/unusable reports in Open-i.
    """
    root = Path(root)
    reports_csv = root / "indiana_reports.csv"
    projections_csv = root / "indiana_projections.csv"
    if not reports_csv.exists() or not projections_csv.exists():
        raise FileNotFoundError(
            f"Cannot find indiana_reports.csv / indiana_projections.csv under {root}"
        )

    reports = _read_reports_csv(reports_csv)
    projections = _read_projections_csv(projections_csv)

    image_root = root / image_subdir
    records: list[dict] = []
    skipped_no_report = 0
    skipped_short = 0
    skipped_no_image = 0

    for entry in projections:
        uid = entry["uid"]
        fname = entry["filename"]
        proj = entry["projection"]

        rep = reports.get(uid)
        if rep is None:
            skipped_no_report += 1
            continue

        caption = _build_caption(rep["findings"], rep["impression"])
        if len(caption) < min_caption_chars:
            skipped_short += 1
            continue

        image_path = image_root / fname
        if not image_path.exists():
            skipped_no_image += 1

        image_id = Path(fname).stem

        records.append({
            "image_id":   image_id,
            "image_path": str(image_path),
            "modality":   "CXR",
            "anatomy":    "chest",
            "caption":    caption,
            "uid":        uid,
            "patient_id": uid,
            "projection": proj,
            "filename":   fname,
            "tasks": {
                "report":           caption,
                "vqa":              [],
                "grounded_caption": None,
                "boxes":            [],
                "masks":            [],
                "K":                None,
            },
            "raw": {
                "mesh":       rep["mesh"],
                "problems":   rep["problems"],
                "indication": _clean(rep["indication"]),
                "comparison": _clean(rep["comparison"]),
            },
        })

    print(f"[indiana] collected {len(records)} image-level records "
          f"(skipped: {skipped_no_report} no-report, "
          f"{skipped_short} too-short, "
          f"{skipped_no_image} missing images on disk)")

    # Three-way patient-level split: 76% train / 4% val / 20% test.
    train, val, test = patient_level_split(
        records, ratios=(0.76, 0.04, 0.20), seed=seed
    )

    for r in train:
        r["split"] = "train"
    for r in val:
        r["split"] = "val"
    for r in test:
        r["split"] = "test"

    dump_json(train, output_train)
    dump_json(val,   output_val)
    dump_json(test,  output_test)
    return records


def convert_indiana(root: str | Path = "data/Chest X"):
    """Public entry point used by ``tools/build_unified_dataset.py``."""
    return convert(root=root)
