"""Split a portion of a dataset's train.json into a separate val.json.

Two modes:

1. **Patient-level split** (default): groups records by ``patient_id`` so that
   all frames of the same patient stay together.  Used for self-owned datasets
   (Indiana, Group-Breast/Thyroid US).

2. **Random split** (``--no-patient-id``): simple random shuffle.  Used for
   public datasets (VQA-RAD, SLAKE) that don't have a ``patient_id`` field.

The original ``<name>_train.json`` is overwritten in-place with the remaining
95 %; the held-out 5 % is written to ``<name>_val.json``.

Usage
-----
    # Self-owned (patient-level, 5 % val from Indiana train)
    python tools/split_val.py --ann data/annotations/indiana_train.json --patient-id-field patient_id

    # Public (random 5 %)
    python tools/split_val.py --ann data/annotations/vqa_train.json
    python tools/split_val.py --ann data/annotations/VQA_train_SLAKE.json
    python tools/split_val.py --ann data/annotations/grounding_train_SLAKE.json

    # All public at once
    python tools/split_val.py --all-public

    # All self-owned at once (requires patient_id in JSON)
    python tools/split_val.py --all-self-owned
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


DEFAULT_VAL_FRACTION = 0.05   # 5 % of train → val
DEFAULT_SEED = 42

PUBLIC_TRAIN_JSONS = [
    "data/annotations/vqa_train.json",
    "data/annotations/VQA_train_SLAKE.json",
    "data/annotations/grounding_train_SLAKE.json",
]

SELF_OWNED_TRAIN_JSONS = [
    "data/annotations/indiana_train.json",
    "data/annotations/group_breast_train.json",
    "data/annotations/group_thyroid_train.json",
]


def _patient_level_split(records: list[dict], val_fraction: float,
                         patient_id_field: str, seed: int) -> tuple[list, list]:
    """Split records so that all frames of one patient stay together."""
    by_pid: dict[str, list[dict]] = {}
    for r in records:
        pid = str(r.get(patient_id_field, r.get("image_id", "")))
        by_pid.setdefault(pid, []).append(r)

    pids = sorted(by_pid.keys())
    rng = random.Random(seed)
    rng.shuffle(pids)
    n_val = max(1, int(round(len(pids) * val_fraction)))
    val_pids = set(pids[:n_val])

    train, val = [], []
    for pid, recs in by_pid.items():
        if pid in val_pids:
            val.extend(recs)
        else:
            train.extend(recs)
    return train, val


def _random_split(records: list[dict], val_fraction: float,
                  seed: int) -> tuple[list, list]:
    """Simple random split (no patient grouping)."""
    n = len(records)
    n_val = max(1, int(round(n * val_fraction)))
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_idx = set(indices[:n_val])
    val = [records[i] for i in sorted(val_idx)]
    train = [records[i] for i in indices if i not in val_idx]
    return train, val


def split_one(ann_path: str | Path,
              val_fraction: float = DEFAULT_VAL_FRACTION,
              seed: int = DEFAULT_SEED,
              patient_id_field: str | None = None) -> tuple[int, int]:
    """Split *ann_path* into train (overwritten in-place) + val (new file).

    Returns ``(n_train, n_val)``.
    """
    ann_path = Path(ann_path)
    if not ann_path.exists():
        print(f"  [skip] {ann_path} not found")
        return 0, 0

    with open(ann_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    if patient_id_field:
        train, val = _patient_level_split(records, val_fraction,
                                          patient_id_field, seed)
    else:
        train, val = _random_split(records, val_fraction, seed)

    # Overwrite train.json with the remaining (1 - val_fraction) portion.
    val_path = ann_path.parent / ann_path.name.replace("_train", "_val")
    with open(ann_path, "w", encoding="utf-8") as f:
        json.dump(train, f, ensure_ascii=False, indent=2)
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val, f, ensure_ascii=False, indent=2)

    mode = "patient-level" if patient_id_field else "random"
    print(f"  {ann_path.name} ({mode}): {len(records)} → "
          f"train={len(train)} + val={len(val)} (→ {val_path.name})")
    return len(train), len(val)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ann", type=str, default=None,
                   help="Path to a single train.json to split.")
    p.add_argument("--all-public", action="store_true",
                   help="Split all public dataset train JSONs (random).")
    p.add_argument("--all-self-owned", action="store_true",
                   help="Split all self-owned dataset train JSONs (patient-level).")
    p.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION,
                   help=f"Fraction of train to hold out as val (default: {DEFAULT_VAL_FRACTION}).")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--patient-id-field", type=str, default="patient_id",
                   help="Field name for patient-level grouping (default: patient_id).")
    args = p.parse_args()

    if not args.ann and not args.all_public and not args.all_self_owned:
        p.error("Pass --ann <path>, --all-public, or --all-self-owned.")

    # Build the list of (path, patient_id_field_or_None) pairs.
    targets: list[tuple[str, str | None]] = []
    if args.all_self_owned:
        for t in SELF_OWNED_TRAIN_JSONS:
            targets.append((t, args.patient_id_field))
    if args.all_public:
        for t in PUBLIC_TRAIN_JSONS:
            targets.append((t, None))
    if args.ann:
        # If --ann is passed, auto-detect patient_id presence.
        targets.append((args.ann, args.patient_id_field))

    print(f"Splitting val (fraction={args.val_fraction}, seed={args.seed}):")
    for path, pid_field in targets:
        split_one(path, val_fraction=args.val_fraction,
                  seed=args.seed, patient_id_field=pid_field)
    print("Done.")


if __name__ == "__main__":
    main()
