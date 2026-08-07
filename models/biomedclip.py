"""BiomedCLIP — biomedical vision-language foundation model (contrastive).

BiomedCLIP is pretrained on PMC-15M (15M biomedical figure-caption pairs)
using contrastive learning with PubMedBERT + ViT.  It has no text generation
capability, so VQA is done via zero-shot answer selection: for each question,
a set of candidate answers is generated based on question patterns, encoded
as text, and the candidate with the highest image-text similarity is returned
as the prediction.

- HuggingFace: ``microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224``
- Paper: BiomedCLIP: a multimodal biomedical foundation model pretrained on
  fifteen million scientific image-text pairs (MICCAI 2023)
- Weight path: ``weights/biomedclip/`` (needs ``open_clip_pytorch_model.bin``
  and ``open_clip_config.json``)

Loading follows the official local-file pattern: ``open_clip`` is used
instead of ``transformers`` because BiomedCLIP ships its config in the
``open_clip`` format (not the standard HF format).
"""

from __future__ import annotations

import json
import os

import torch
from PIL import Image
from rich import print

from . import BaseLM


# --------------------------------------------------------------------------- #
# Candidate-answer generation by question type
# --------------------------------------------------------------------------- #
# BiomedCLIP cannot generate free-form text.  For VQA we construct a set of
# candidate answers based on the question's linguistic pattern, then select
# the one with the highest image-text similarity.  This mirrors the official
# BiomedCLIP VQA evaluation protocol (multiple-choice scoring).

_YES_NO_PATTERNS = [
    "is there", "are there", "does", "do ", "can ", "could",
    "has", "have", "should", "was", "were", "is this", "are these",
]

_MODALITY_CANDIDATES = ["X-ray", "CT", "MRI", "ultrasound", "mammography", "endoscopy"]

_ORGAN_CANDIDATES = [
    "lung", "heart", "liver", "kidney", "spleen", "pancreas",
    "brain", "stomach", "colon", "bladder", "chest", "abdomen",
    "spine", "rib", "diaphragm", "mediastinum", "pleura",
]

_PLANE_CANDIDATES = ["PA", "AP", "lateral", "axial", "coronal", "sagittal"]

_COUNT_CANDIDATES = ["0", "1", "2", "3", "4", "5"]

_ABNORMALITY_CANDIDATES = [
    "normal", "abnormal", "pneumonia", "effusion", "nodule", "mass",
    "atelectasis", "cardiomegaly", "pneumothorax", "consolidation",
    "edema", "fracture", "lesion", "tumor", "calcification",
    "opacity", "thickening", "hernia", "no finding",
]

_PROCEDURE_CANDIDATES = [
    "colonoscopy", "endoscopy", "X-ray", "CT scan", "MRI scan",
    "ultrasound", "mammography",
]


def _generate_candidates(question: str) -> list[str]:
    """Generate candidate answers based on question patterns."""
    q_lower = question.lower().strip()

    # Yes/No questions
    if any(q_lower.startswith(p) or f" {p}" in q_lower for p in _YES_NO_PATTERNS):
        if "not" in q_lower or "n't" in q_lower:
            return ["yes", "no"]
        return ["yes", "no"]

    # Modality questions
    if "modality" in q_lower or "imaging" in q_lower or "type of" in q_lower:
        return _MODALITY_CANDIDATES

    # Plane questions
    if "plane" in q_lower or "view" in q_lower:
        return _PLANE_CANDIDATES

    # Organ / body part questions
    if "organ" in q_lower or "body part" in q_lower or "where" in q_lower:
        return _ORGAN_CANDIDATES

    # Counting questions
    if "how many" in q_lower or "number of" in q_lower:
        return _COUNT_CANDIDATES

    # Abnormality / disease questions
    if "abnormal" in q_lower or "disease" in q_lower or "finding" in q_lower \
       or "pathology" in q_lower or "lesion" in q_lower or "tumor" in q_lower:
        return _ABNORMALITY_CANDIDATES

    # Procedure questions
    if "procedure" in q_lower or "exam" in q_lower or "study" in q_lower:
        return _PROCEDURE_CANDIDATES

    # Default: return common short answers
    return _ABNORMALITY_CANDIDATES + _ORGAN_CANDIDATES + ["yes", "no", "normal", "unknown"]


class BiomedCLIP(BaseLM):
    """BiomedCLIP wrapper for VQA via zero-shot answer selection.

    Since BiomedCLIP is a contrastive model (no generation), VQA is done by:
    1. Encoding the image
    2. Encoding ``"{question} {candidate}"`` for each candidate answer
    3. Returning the candidate with the highest image-text cosine similarity

    Loading follows the official local-file pattern using ``open_clip``:
    - Reads ``open_clip_config.json`` from the weight directory
    - Registers a local model name with ``_MODEL_CONFIGS``
    - Creates model + transforms via ``create_model_and_transforms``
    - Tokenizes with ``get_tokenizer`` (PubMedBERT, context_length=256)
    """

    # Context length for PubMedBERT tokenizer (matches official example).
    CONTEXT_LENGTH = 256

    def __init__(self, name: str = "BiomedCLIP", device: str = "cuda",
                 dtype=torch.float32):
        super().__init__(name=name, device=device, dtype=dtype)
        checkpoint_dir = "weights/biomedclip"
        print(f"=> Loading BiomedCLIP model ({checkpoint_dir})")

        from open_clip import create_model_and_transforms, get_tokenizer
        from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS

        # ---- Load open_clip_config.json ----
        config_path = os.path.join(checkpoint_dir, "open_clip_config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        model_cfg = config["model_cfg"]
        preprocess_cfg = config["preprocess_cfg"]

        # ---- Register local model name ----
        local_model_name = "biomedclip_local"
        if (not local_model_name.startswith(HF_HUB_PREFIX)
                and local_model_name not in _MODEL_CONFIGS):
            _MODEL_CONFIGS[local_model_name] = model_cfg

        # ---- Tokenizer ----
        self.tokenizer = get_tokenizer(local_model_name)

        # ---- Model + image preprocessing ----
        weights_path = os.path.join(checkpoint_dir, "open_clip_pytorch_model.bin")
        self.model, _, self.preprocess = create_model_and_transforms(
            model_name=local_model_name,
            pretrained=weights_path,
            **{f"image_{k}": v for k, v in preprocess_cfg.items()},
        )
        self.model = self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def _encode_image(self, image_path: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a single image to (features, logit_scale)."""
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        # open_clip model forward returns (image_features, text_features, logit_scale)
        # when called with both; for image-only we call encode_image directly.
        image_features = self.model.encode_image(image_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.model.logit_scale.exp()
        return image_features, logit_scale  # (1, D), scalar

    @torch.no_grad()
    def _encode_texts(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of text strings to (features, logit_scale)."""
        tokens = self.tokenizer(texts, context_length=self.CONTEXT_LENGTH).to(self.device)
        text_features = self.model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = self.model.logit_scale.exp()
        return text_features, logit_scale  # (N, D), scalar

    @torch.no_grad()
    def generate(self, paths, prompt, **kwargs):
        """Zero-shot VQA via image-text similarity.

        Parameters
        ----------
        paths : str or list[str] — image file path(s); only the first is used.
        prompt : str — the question text (may include a ``[vqa]`` prefix).

        Returns
        -------
        str — the best-matching candidate answer.
        """
        if isinstance(paths, str):
            paths = paths.split("|")
        image_path = paths[0]

        # Strip [vqa] prefix if present
        question = prompt.strip()
        if question.startswith("[vqa]"):
            question = question[len("[vqa]"):].strip()

        # Generate candidate answers based on question type
        candidates = _generate_candidates(question)

        # Encode image and candidate texts
        img_feat, logit_scale = self._encode_image(image_path)     # (1, D)
        text_inputs = [f"{question} {c}" for c in candidates]
        txt_feat, _ = self._encode_texts(text_inputs)               # (N, D)

        # Compute scaled cosine similarity and pick the best
        logits = (logit_scale * img_feat @ txt_feat.T).squeeze(0)  # (N,)
        best_idx = logits.argmax().item()
        return candidates[best_idx]
