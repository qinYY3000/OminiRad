import ast
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
# Datasets that previously lived on the "legacy" MiniGPT-Med code path and
# must keep importing ``add_default_structured_fields`` so they are mixable
# with the unified-schema datasets via ConcatDataset.
LEGACY_DATASETS = [
    REPO_ROOT / "minigpt4" / "datasets" / "datasets" / "indiana_dataset.py",
    REPO_ROOT / "minigpt4" / "datasets" / "datasets" / "radvqa_dataset.py",
    REPO_ROOT / "minigpt4" / "datasets" / "datasets" / "nlst_dataset.py",
    REPO_ROOT / "minigpt4" / "datasets" / "datasets" / "rsna_dataset.py",
    REPO_ROOT / "minigpt4" / "datasets" / "datasets" / "SLAKE_dataset.py",
]
US_DATASET = REPO_ROOT / "minigpt4" / "datasets" / "datasets" / "unified_us_dataset.py"
BASE_DATASET = REPO_ROOT / "minigpt4" / "datasets" / "datasets" / "base_dataset.py"
STRUCTURED_FIELDS = REPO_ROOT / "minigpt4" / "datasets" / "datasets" / "structured_fields.py"


class UnifiedSchemaTests(unittest.TestCase):
    def test_structured_fields_utility_exists(self):
        self.assertTrue(STRUCTURED_FIELDS.exists())

    def test_legacy_datasets_import_structured_fields(self):
        for path in LEGACY_DATASETS:
            text = path.read_text(encoding="utf-8")
            self.assertIn("add_default_structured_fields", text,
                          f"{path.name} must import add_default_structured_fields")

    def test_unified_us_dataset_loads_masks_as_tensors(self):
        text = US_DATASET.read_text(encoding="utf-8")
        self.assertIn("_load_masks", text)
        self.assertIn("torch.from_numpy", text)
        # masks field must appear in both __getitem__ and collater
        self.assertGreaterEqual(text.count('"masks"'), 2)

    def test_concat_dataset_collater_uses_union(self):
        text = BASE_DATASET.read_text(encoding="utf-8")
        self.assertNotIn("shared_keys = all_keys", text)
        self.assertIn("union", text.lower())

    def test_mimic_dataset_files_removed(self):
        """MIMIC has been removed from the project; make sure no stragglers remain."""
        leftovers = [
            REPO_ROOT / "minigpt4" / "datasets" / "datasets" / "mimic_cxr_dataset.py",
            REPO_ROOT / "minigpt4" / "configs" / "datasets" / "mimic_cxr",
            REPO_ROOT / "minigpt4" / "configs" / "datasets" / "detect_mimic",
            REPO_ROOT / "data" / "annotations" / "MIMIC_train.json",
            REPO_ROOT / "data" / "annotations" / "MIMIC_test.json",
        ]
        still_there = [str(p) for p in leftovers if p.exists()]
        self.assertEqual(still_there, [],
                         f"These MIMIC artifacts should have been removed: {still_there}")


if __name__ == "__main__":
    unittest.main()
