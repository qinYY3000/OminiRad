"""Static + functional tests for the Indiana University Chest-Xray pipeline."""

import ast
import json
import os
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONV_FILE      = REPO_ROOT / "tools" / "converters" / "indiana.py"
DATASET_FILE   = REPO_ROOT / "minigpt4" / "datasets" / "datasets" / "indiana_dataset.py"
BUILDER_FILE   = REPO_ROOT / "minigpt4" / "datasets" / "builders" / "image_text_pair_builder.py"
BUILDER_INIT   = REPO_ROOT / "minigpt4" / "datasets" / "builders" / "__init__.py"
TRAIN_YAML     = REPO_ROOT / "train_configs" / "omnirad_finetune.yaml"
EVAL_YAML      = REPO_ROOT / "eval_configs" / "omnirad_evaluation.yaml"
ANN_TRAIN      = REPO_ROOT / "data" / "annotations" / "indiana_train.json"


class IndianaStaticTests(unittest.TestCase):
    def test_converter_module_exists(self):
        self.assertTrue(CONV_FILE.exists(),
                        f"Missing converter: {CONV_FILE}")

    def test_dataset_module_exists(self):
        self.assertTrue(DATASET_FILE.exists(),
                        f"Missing dataset module: {DATASET_FILE}")

    def test_builder_registered(self):
        text = BUILDER_FILE.read_text(encoding="utf-8")
        self.assertIn('register_builder("indiana_cxr")', text)
        self.assertIn("IndianaCXRDataset", text)

    def test_builder_exported(self):
        text = BUILDER_INIT.read_text(encoding="utf-8")
        self.assertIn("IndianaCxrBuilder", text)

    def test_train_yaml_includes_indiana(self):
        text = TRAIN_YAML.read_text(encoding="utf-8")
        self.assertIn("indiana_cxr:", text)

    def test_eval_yaml_includes_indiana(self):
        text = EVAL_YAML.read_text(encoding="utf-8")
        self.assertIn("indiana_cxr:", text)

    def test_dataset_overrides_load_image_for_png(self):
        """IndianaCXRDataset must read .png files, not .jpg."""
        text = DATASET_FILE.read_text(encoding="utf-8")
        # The dataset must reference the .png extension.
        self.assertIn(".png", text)
        # And it must declare a self-contained Dataset (no MIMIC inheritance).
        self.assertIn("class IndianaCXRDataset", text)
        self.assertNotIn("MimicCxrDataset", text,
                         "IndianaCXRDataset must not depend on the removed MIMIC dataset")

    def test_converter_redacts_xxxx_tokens(self):
        """The Open-i ``XXXX`` PHI placeholder should not leak into captions."""
        # Parse the module statically — we just want to make sure
        # there's a substitution/replace targeting "XXXX".
        text = CONV_FILE.read_text(encoding="utf-8")
        self.assertIn("XXXX", text, "converter must handle the XXXX placeholder")
        self.assertIn("REDACTED", text,
                      "converter must rewrite XXXX into a stable token")


class IndianaJSONShapeTests(unittest.TestCase):
    """Verify the JSON produced by the converter has the right shape.

    Skipped automatically when the JSON has not been built yet (the user must
    run ``python tools/build_unified_dataset.py --datasets indiana`` first).
    """

    def setUp(self):
        if not ANN_TRAIN.exists():
            self.skipTest(
                f"{ANN_TRAIN} not built yet; run the converter first."
            )
        with open(ANN_TRAIN, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def test_records_are_nonempty(self):
        self.assertGreater(len(self.data), 0)

    def test_required_fields_present(self):
        sample = self.data[0]
        for key in ("image_id", "caption", "modality", "anatomy"):
            self.assertIn(key, sample, f"missing required key: {key}")
        self.assertEqual(sample["modality"], "CXR")
        self.assertEqual(sample["anatomy"], "chest")

    def test_image_id_has_no_png_suffix(self):
        """``image_id`` must not include the .png suffix — it is appended at load time."""
        for sample in self.data[:50]:
            self.assertFalse(sample["image_id"].endswith(".png"),
                             f"image_id should not contain .png: {sample['image_id']}")

    def test_no_raw_xxxx_in_captions(self):
        """Captions must have been redacted (no bare ``XXXX`` tokens)."""
        leak = 0
        for sample in self.data[:200]:
            if " XXXX " in sample["caption"]:
                leak += 1
        self.assertEqual(leak, 0,
                         f"{leak} captions still contain raw XXXX tokens")


if __name__ == "__main__":
    unittest.main()
