"""Indiana University Chest X-Ray (Open-i) dataset.

Report-generation dataset built from the two CSV files distributed by NIH
Open-i (Indiana University), converted to the unified-schema JSON by
``tools/converters/indiana.py``.

Each JSON record has the shape::

    {
        "image_id": "<uid>_IM-XXXX-YYYY.dcm",   # no ".png" suffix
        "caption":  "<findings>. Impression: <impression>",
        "modality": "CXR",
        "anatomy":  "chest",
        ...
    }

The dataset class loads the corresponding ``<image_id>.png`` file from the
``images/`` directory and applies the same ``vis_processor`` / instruction
template that other report-generation datasets use.
"""

from __future__ import annotations

import json
import os
import random

from PIL import Image
from torch.utils.data import Dataset

from minigpt4.datasets.datasets.structured_fields import add_default_structured_fields


# Open-i frames are named ``<uid>_IM-XXXX-YYYY.dcm.png``.  ``image_id`` in the
# JSON drops the trailing ``.png`` (mirroring how MIMIC stored ``.jpg`` IDs),
# so we re-append the extension at load time.
_IMAGE_EXT = ".png"


# Re-used as the prompt pool for both training and evaluation so that the
# evaluator and the trained model see consistent instruction wording.
INDIANA_INSTRUCTION_POOL = [
    "Describe this image in detail",
    "Take a look at this image and describe what you notice",
    "Please provide a detailed description of the picture",
    "Could you describe the contents of this image for me?",
]


class IndianaCXRDataset(Dataset):
    """Indiana Chest-X-ray dataset for report generation."""

    def __init__(self, vis_processor, text_processor, vis_root, ann_path):
        self.vis_root = vis_root
        self.vis_processor = vis_processor
        self.text_processor = text_processor

        with open(ann_path, "r", encoding="utf-8") as f:
            self.ann = json.load(f)

        self.instruction_pool = list(INDIANA_INSTRUCTION_POOL)

    def __len__(self):
        return len(self.ann)

    def load_image(self, image_id):
        image_file = f"{image_id}{_IMAGE_EXT}"
        image_path = os.path.join(self.vis_root, image_file)
        grayscale_image = Image.open(image_path).convert("L")
        image = Image.new("RGB", grayscale_image.size)
        image.paste(grayscale_image)
        return self.vis_processor(image)

    def __getitem__(self, index):
        info = self.ann[index]
        image = self.load_image(info["image_id"])
        instruction = random.choice(self.instruction_pool)
        instruction = (f"[INST] <Img><ImageHere></Img> "
                       f"{self.text_processor(instruction)} [/INST]")

        return add_default_structured_fields({
            "image": image,
            "instruction_input": instruction,
            "answer": info["caption"],
            "image_id": info["image_id"],
        }, modality="CXR", anatomy="chest")


# ---------------------------------------------------------------------------
# Evaluation-time dataset
# ---------------------------------------------------------------------------
class evalIndianaCXRDataset(Dataset):
    """Eval pair (image, question, image_id) for the Indiana test split."""

    def __init__(self, loaded_data, vis_processor, root_path):
        self.loaded_data = loaded_data
        self.root_path = root_path
        self.vis_processor = vis_processor
        self.instruction_pool = list(INDIANA_INSTRUCTION_POOL)

    def __len__(self):
        return len(self.loaded_data)

    def __getitem__(self, idx):
        info = self.loaded_data[idx]
        img_id = f"{info['image_id']}{_IMAGE_EXT}"
        image_path = os.path.join(self.root_path, img_id)
        grayscale_image = Image.open(image_path).convert("L")
        image = Image.new("RGB", grayscale_image.size)
        image.paste(grayscale_image)
        image = self.vis_processor(image)

        question = random.choice(self.instruction_pool)
        return image, question, img_id
