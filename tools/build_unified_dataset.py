"""Build all unified-schema JSON annotations from raw datasets.

Usage examples
==============
# Convert everything (assumes all raw data is in place under data/)
python tools/build_unified_dataset.py --all

# Convert only the private US datasets
python tools/build_unified_dataset.py --datasets group_breast group_thyroid

# Convert the Indiana University Open-i Chest-Xray dataset
python tools/build_unified_dataset.py --datasets indiana

# Show what would be done without writing files
python tools/build_unified_dataset.py --all --dry-run

Outputs go to: data/annotations/<dataset>_<split>.json
See docs/dataset_plan.md §3 for the unified schema and §4 for the on-disk layout.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


# Allow `python tools/build_unified_dataset.py` to import tools.*
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.converters import group_us, public, indiana   # noqa: E402


# -----------------------------------------------------------------------------
# All available dataset keys (matches `--datasets` arg).
# MIMIC and NLST are intentionally absent — those datasets are no longer used
# (MIMIC dropped, NLST raw images unavailable).
# -----------------------------------------------------------------------------
ALL_DATASETS = [
    "radvqa", "slake", "rsna",
    "group_breast", "group_thyroid",
    "indiana",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Build unified-schema JSON annotations for OmniRad.")
    p.add_argument("--all", action="store_true",
                   help="Convert all datasets listed in ALL_DATASETS.")
    p.add_argument("--datasets", nargs="+", choices=ALL_DATASETS, default=[],
                   help=f"Subset of datasets to convert. Choices: {ALL_DATASETS}")
    p.add_argument("--dry-run", action="store_true",
                   help="Print plan only; do not write files.")

    # Source paths for the inherited public datasets (their JSON files)
    p.add_argument("--radvqa-src",  type=str,
                   default="/miniGPT-Med/json_files/SLAKE/eecv24/mearged_merged_VQA_data.json")
    p.add_argument("--slake-src",   type=str,
                   default="/miniGPT-Med/json_files/SLAKE/grounding_train_SLAKE.json")
    p.add_argument("--rsna-src",    type=str,
                   default="/miniGPT-Med/json_files/rsna/RSNA_test.json")

    # Image roots for inherited public datasets
    p.add_argument("--radvqa-img-root", type=str, default="data/radvqa/imgs")
    p.add_argument("--slake-img-root",  type=str, default="data/slake/imgs")
    p.add_argument("--rsna-img-root",   type=str, default="data/rsna/RSNA-bbox-1024")

    # Root paths for private US datasets
    p.add_argument("--breast-root",  type=str, default="data/group_breast")
    p.add_argument("--thyroid-root", type=str, default="data/group_thyroid")

    # Root path for the Indiana University Open-i dataset
    p.add_argument("--indiana-root", type=str, default="data/Chest X")

    return p.parse_args()


def maybe_run(name: str, args, todo: set):
    """Print a plan line; if not dry-run, dispatch to the right converter."""
    if name not in todo:
        return
    t0 = time.time()
    print(f"\n========== [{name}] ==========")
    try:
        if args.dry_run:
            print(f"  (dry-run) would convert {name}")
            return
        if   name == "radvqa":
            public.convert_radvqa(args.radvqa_src, args.radvqa_img_root)
        elif name == "slake":
            public.convert_slake(args.slake_src, args.slake_img_root)
        elif name == "rsna":
            public.convert_rsna(args.rsna_src, args.rsna_img_root)
        elif name == "group_breast":
            group_us.convert_breast(args.breast_root)
        elif name == "group_thyroid":
            group_us.convert_thyroid(args.thyroid_root)
        elif name == "indiana":
            indiana.convert_indiana(args.indiana_root)
        else:
            print(f"  [skip] unknown dataset key '{name}'")
    except FileNotFoundError as e:
        print(f"  [warn] source not found, skipping: {e}")
    except Exception as e:
        print(f"  [error] {name}: {type(e).__name__}: {e}")
        raise
    print(f"  done in {time.time() - t0:.1f}s")


def main():
    args = parse_args()

    if args.all and args.datasets:
        sys.exit("Use either --all or --datasets, not both.")
    todo = set(ALL_DATASETS) if args.all else set(args.datasets)
    if not todo:
        sys.exit("Nothing to do. Pass --all or --datasets X Y Z.")

    print(f"OmniRad dataset builder: will process {sorted(todo)}")
    if args.dry_run:
        print("(dry run — no files will be written)")

    Path("data/annotations").mkdir(parents=True, exist_ok=True)

    for name in ALL_DATASETS:
        maybe_run(name, args, todo)

    print("\nAll done.")


if __name__ == "__main__":
    main()
