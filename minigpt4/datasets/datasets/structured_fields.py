"""Utility to add default structured-supervision fields to legacy dataset samples.

Legacy MiniGPT-Med datasets (IndianaCXRDataset, RadVQADataset, NlstDataset, etc.)
only return ``{image, instruction_input, answer, image_id}``.  When they are
mixed with ``UnifiedUSDataset`` in a ``ConcatDataset``, the default
``ConcatDataset.collater`` performs a key-intersection and silently drops all
structured fields (boxes, masks, K, ...).

Calling ``add_default_structured_fields(sample)`` on every legacy sample ensures
a unified schema so that no fields are lost during concatenation.
"""

from __future__ import annotations

import torch


def add_default_structured_fields(
    sample: dict,
    *,
    modality: str = "unknown",
    anatomy: str = "unknown",
) -> dict:
    """Return *sample* with OmniRad structured-supervision fields guaranteed.

    If the field already exists it is left untouched; otherwise a sensible
    "empty" default is inserted so that downstream collaters and the model's
    ``extract_structured_targets`` always see the same keys.
    """
    defaults: dict = {
        "modality": modality,
        "anatomy": anatomy,
        "boxes": torch.zeros((0, 4), dtype=torch.float32),
        "box_scales": [],
        "anatomy_regions": [],
        "mask_paths": [],
        "masks": [],
        "K": torch.tensor(0, dtype=torch.long),
        "has_structured_supervision": torch.tensor(False, dtype=torch.bool),
        "raw_tasks": {},
    }
    for key, default_val in defaults.items():
        sample.setdefault(key, default_val)
    return sample
