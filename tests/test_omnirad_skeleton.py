import ast
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_FILE = REPO_ROOT / "minigpt4" / "models" / "omnirad.py"
MODEL_INIT_FILE = REPO_ROOT / "minigpt4" / "models" / "__init__.py"
MODEL_CONFIG_FILE = REPO_ROOT / "minigpt4" / "configs" / "models" / "omnirad.yaml"


class OmniRadSkeletonTests(unittest.TestCase):
    def test_omnirad_model_file_exists(self):
        self.assertTrue(MODEL_FILE.exists(), f"Missing model file: {MODEL_FILE}")

    def test_omnirad_default_model_config_exists(self):
        self.assertTrue(MODEL_CONFIG_FILE.exists(), f"Missing config file: {MODEL_CONFIG_FILE}")

    def test_omnirad_is_exported_from_models_package(self):
        init_text = MODEL_INIT_FILE.read_text(encoding="utf-8")
        self.assertIn("from minigpt4.models.omnirad import OmniRad", init_text)
        self.assertIn('"OmniRad"', init_text)

    def test_omnirad_class_declares_pretrained_config(self):
        module = ast.parse(MODEL_FILE.read_text(encoding="utf-8"))
        omnirad_cls = None
        for node in module.body:
            if isinstance(node, ast.ClassDef) and node.name == "OmniRad":
                omnirad_cls = node
                break
        self.assertIsNotNone(omnirad_cls, "Class `OmniRad` not found")

        config_assign = None
        for stmt in omnirad_cls.body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name) and target.id == "PRETRAINED_MODEL_CONFIG_DICT":
                        config_assign = stmt.value
                        break
        self.assertIsNotNone(config_assign, "`PRETRAINED_MODEL_CONFIG_DICT` not declared")
        self.assertIsInstance(config_assign, ast.Dict)

        keys = []
        values = []
        for key, value in zip(config_assign.keys, config_assign.values):
            if isinstance(key, ast.Constant):
                keys.append(key.value)
            if isinstance(value, ast.Constant):
                values.append(value.value)
        self.assertIn("pretrain", keys)
        self.assertIn("configs/models/omnirad.yaml", values)


if __name__ == "__main__":
    unittest.main()
