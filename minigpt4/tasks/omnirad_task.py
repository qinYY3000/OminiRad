"""
OmniRad evaluation task — supports multi-task validation during training.

Extends ImageTextPretrainTask with:
- valid_step: calls model.generate(), collects structured outputs
- evaluation: runs valid_step over the full dataloader
- after_evaluation: computes per-task metrics (cardinality accuracy)
"""

from __future__ import annotations

import logging
import os

import torch
from minigpt4.common.registry import registry
from minigpt4.common.dist_utils import get_rank
from minigpt4.tasks.image_text_pretrain import ImageTextPretrainTask
from tqdm import tqdm


@registry.register_task("omnirad_eval")
class OmniRadTask(ImageTextPretrainTask):
    """Task with validation support for OmniRad multi-task outputs."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._val_results = []

    def valid_step(self, model, samples):
        """Run inference on a validation batch and collect results."""
        images = samples.get("image")
        instructions = samples.get("instruction_input", [])
        image_ids = samples.get("image_id", [])

        if images is None or not instructions:
            return []

        # Build conversation texts
        texts = instructions if isinstance(instructions, list) else [instructions]

        # Generate with OmniRad's structured generate
        results = model.generate(
            images,
            texts,
            max_new_tokens=200,
            do_sample=False,
        )

        output = []
        for idx, result in enumerate(results):
            entry = {
                "image_id": image_ids[idx] if idx < len(image_ids) else str(idx),
                "text": result.get("text", "") if isinstance(result, dict) else str(result),
                "seg_count": result.get("seg_count", 0) if isinstance(result, dict) else 0,
                "box_tokens": result.get("box_tokens", []) if isinstance(result, dict) else [],
                "has_loc": result.get("has_loc", False) if isinstance(result, dict) else False,
                "gt_answer": samples.get("answer", [""] * len(results))[idx] if idx < len(samples.get("answer", [])) else "",
                "gt_K": samples.get("K", [0] * len(results))[idx].item() if hasattr(samples.get("K", [0]), '__getitem__') and idx < len(samples.get("K", [])) and hasattr(samples["K"][idx], 'item') else 0,
            }
            output.append(entry)

        return output

    @torch.no_grad()
    def evaluation(self, model, data_loader):
        """Run validation over the entire dataloader, collecting structured results."""
        model.eval()
        all_results = []
        for samples in tqdm(data_loader, desc="Validating", disable=get_rank() != 0):
            # Move tensors to CUDA
            if torch.cuda.is_available():
                for k, v in samples.items():
                    if isinstance(v, torch.Tensor):
                        samples[k] = v.cuda()
            batch_results = self.valid_step(model, samples)
            all_results.extend(batch_results)
        return all_results

    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        """Compute aggregate metrics from validation results."""
        n_total = len(val_result)
        if n_total == 0:
            return {"agg_metrics": 0.0, "n_samples": 0}

        # Count correct cardinality predictions (predicted K == GT K)
        n_correct_k = sum(
            1 for r in val_result
            if r.get("seg_count", 0) == r.get("gt_K", 0)
        )
        cardinality_acc = n_correct_k / n_total

        # Count how many results have structured output
        n_structured = sum(1 for r in val_result if r.get("seg_count", 0) > 0 or r.get("box_tokens"))

        logging.info(
            "[OmniRad Eval] epoch=%d samples=%d cardinality_acc=%.4f structured=%d",
            epoch, n_total, cardinality_acc, n_structured,
        )

        return {
            "agg_metrics": cardinality_acc,
            "n_samples": n_total,
            "cardinality_acc": cardinality_acc,
            "n_structured": n_structured,
        }
