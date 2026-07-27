import ast
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_FILE = REPO_ROOT / "minigpt4" / "datasets" / "datasets" / "unified_us_dataset.py"
BUILDER_FILE = REPO_ROOT / "minigpt4" / "datasets" / "builders" / "image_text_pair_builder.py"
BUILDER_INIT_FILE = REPO_ROOT / "minigpt4" / "datasets" / "builders" / "__init__.py"
MODEL_FILE = REPO_ROOT / "minigpt4" / "models" / "omnirad.py"


class StructuredSupervisionStaticTests(unittest.TestCase):
    def test_unified_us_dataset_file_exists(self):
        self.assertTrue(DATASET_FILE.exists(), f"Missing dataset file: {DATASET_FILE}")

    def test_group_us_builders_are_registered(self):
        text = BUILDER_FILE.read_text(encoding="utf-8")
        self.assertIn('register_builder("group_breast_us")', text)
        self.assertIn('register_builder("group_thyroid_us")', text)
        self.assertIn("UnifiedUSDataset", text)

    def test_group_us_builders_are_exported(self):
        text = BUILDER_INIT_FILE.read_text(encoding="utf-8")
        self.assertIn("GroupBreastUSBuilder", text)
        self.assertIn("GroupThyroidUSBuilder", text)

    def test_unified_us_dataset_returns_structured_fields(self):
        module = ast.parse(DATASET_FILE.read_text(encoding="utf-8"))
        constants = {
            node.value
            for node in ast.walk(module)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        for field in [
            "boxes",
            "box_scales",
            "anatomy_regions",
            "mask_paths",
            "K",
            "has_structured_supervision",
        ]:
            self.assertIn(field, constants)

    def test_omnirad_forward_extracts_structured_targets(self):
        text = MODEL_FILE.read_text(encoding="utf-8")
        self.assertIn("extract_structured_targets", text)
        self.assertIn("structured_targets", text)
        self.assertIn("has_structured_supervision", text)
        self.assertIn('"boxes"', text)
        self.assertIn('"mask_paths"', text)

    def test_unified_us_dataset_supports_five_tasks(self):
        """User-spec: ultrasound datasets must cover
        segmentation / detection / report / refer / identify (5 tasks)."""
        text = DATASET_FILE.read_text(encoding="utf-8")
        for task in ["report", "segmentation", "detection", "refer", "identify"]:
            self.assertIn(f'"{task}"', text,
                          f"task_pool missing '{task}' in UnifiedUSDataset")
        # identify branch must have its own answer/prompt builder
        self.assertIn("_identify_prompt_answer", text)
        self.assertIn("[identify]", text)

    def test_omnirad_generate_supports_return_masks(self):
        """OmniRad.generate must expose ``return_masks`` so the eval pipeline
        can request mask outputs and run Dice on the segmentation task."""
        text = MODEL_FILE.read_text(encoding="utf-8")
        self.assertIn("return_masks", text,
                      "OmniRadModel.generate must accept a return_masks kwarg")
        self.assertIn("_decode_masks_from_outputs", text,
                      "Mask decoding helper must exist on OmniRadModel")
        # mask path goes through the SEG hidden state → projector → mask_decoder
        self.assertIn("seg_projector", text)
        self.assertIn("mask_decoder(", text)


if __name__ == "__main__":
    unittest.main()
