"""Build group_breast report-generation annotations for API baseline eval.

The breast-ultrasound dataset has this layout:
    data/group_breast/
        frames/<CASE>/<frame>.jpg      (one or more ultrasound frames per case)
        reports/<CASE>/s1.txt          (one English report per case)

This script pairs the first frame of each case with its report and writes
the ``data/annotations/group_breast_test.json`` expected by
``eval_scripts/api_baseline_generate.py --dataset group_breast_us --task report``.

Placeholder reports ("This is the content for the English report.") are
skipped and counted.

Usage:
    python tools/build_group_breast_annotations.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRAMES_DIR = ROOT / "data" / "group_breast" / "frames"
REPORTS_DIR = ROOT / "data" / "group_breast" / "reports"
OUT = ROOT / "data" / "annotations" / "group_breast_test.json"

PLACEHOLDER_RE = re.compile(r"this is the content|^\s*$", re.IGNORECASE)


def iter_reports():
    """Yield (case_id, text) for every real (non-placeholder) report."""
    n_total, n_skip = 0, 0
    for txt in sorted(REPORTS_DIR.glob("*/*.txt")):
        n_total += 1
        text = txt.read_text(encoding="utf-8").strip()
        if len(text) < 50 or PLACEHOLDER_RE.search(text):
            n_skip += 1
            continue
        yield txt.parent.name, text
    print(f"Reports: {n_total} total, {n_skip} placeholder/short, "
          f"{n_total - n_skip} usable")


def main():
    anns = []
    used = 0
    for case_id, report_text in iter_reports():
        frames = sorted((FRAMES_DIR / case_id).glob("*.jpg"))
        if not frames:
            print(f"[skip] {case_id}: no frames")
            continue
        # Use the first frame as the representative image
        rel = frames[0].relative_to(ROOT).as_posix()
        anns.append({
            "image_id": case_id,
            "image_path": rel,
            "modality": "ultrasound",
            "tasks": {"report": report_text},
        })
        used += 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(anns, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"Wrote {used} annotations to {OUT}")


if __name__ == "__main__":
    main()
