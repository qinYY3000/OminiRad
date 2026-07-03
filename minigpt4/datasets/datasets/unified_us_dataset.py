from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate


class UnifiedUSDataset(Dataset):
    """Unified-schema ultrasound dataset with structured supervision.

    This dataset consumes the JSON files produced by `tools/build_unified_dataset.py`
    for Group-Breast US and Group-Thyroid US. In addition to the standard
    MiniGPT-Med fields (`image`, `instruction_input`, `answer`, `image_id`), each
    sample carries variable-length supervision for OmniRad's dense heads:
    boxes, box scales, anatomy regions, mask paths, and K.

    Per-sample task is randomly drawn from the pool below (each call to
    ``__getitem__`` returns one task view of the frame):

        ``report``       — generate the full ultrasound report
        ``segmentation`` — emit K ``<SEG>`` tokens, one per lesion
        ``detection``    — emit ``<BOX_S/M/L>`` tokens with scale supervision
        ``refer``        — produce a ``<LOC>`` token grounded on anatomy region
        ``identify``     — given a bbox, name the lesion class & region
        ``vqa``          — when ``tasks.vqa`` is non-empty, sample a QA pair
    """

    def __init__(self, vis_processor, text_processor, vis_root: str, ann_path: str,
                 anatomy: str | None = None):
        self.vis_root = vis_root
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.anatomy = anatomy

        with open(ann_path, "r", encoding="utf-8") as f:
            ann = json.load(f)
        if isinstance(ann, dict):
            ann = ann.get("annotations", ann.get("data", []))
        self.ann = ann

        self.task_pool = ["report", "segmentation", "detection", "refer", "identify"]

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        info = self.ann[index]
        tasks = info.get("tasks", {}) or {}
        image = self._load_image(info)
        boxes = self._boxes_to_tensor(tasks.get("boxes", []))
        box_scales = [box.get("scale", "M") for box in tasks.get("boxes", [])]
        anatomy_regions = [box.get("anatomy_region") for box in tasks.get("boxes", [])]
        mask_paths_raw = [mask.get("mask_path") for mask in tasks.get("masks", []) if mask.get("mask_path")]
        masks = self._load_masks(mask_paths_raw)
        K = int(tasks.get("K", len(masks) if masks else len(box_scales)))

        instruction, answer = self._build_instruction_answer(info, tasks, K, box_scales, anatomy_regions)

        return {
            "image": image,
            "instruction_input": instruction,
            "answer": answer,
            "image_id": info.get("image_id", str(index)),
            "modality": info.get("modality", "ultrasound"),
            "anatomy": info.get("anatomy", self.anatomy),
            "boxes": boxes,
            "box_scales": box_scales,
            "anatomy_regions": anatomy_regions,
            "mask_paths": mask_paths_raw,
            "masks": masks,
            "K": torch.tensor(K, dtype=torch.long),
            "has_structured_supervision": torch.tensor(True, dtype=torch.bool),
            "raw_tasks": tasks,
        }

    def _load_image(self, info: dict) -> torch.Tensor:
        image_path = self._resolve_path(info.get("image_path", ""))
        grayscale_image = Image.open(image_path).convert("L")
        image = Image.new("RGB", grayscale_image.size)
        image.paste(grayscale_image)
        return self.vis_processor(image)

    def _load_masks(self, mask_paths: list[str]) -> list[torch.Tensor]:
        """Load binary mask files as float tensors (1=foreground, 0=background).

        Returns an empty list when no mask paths are provided.  Each returned
        tensor has shape ``(H, W)`` in float32.  Masks are NOT resized here —
        the model's mask decoder handles resolution matching.
        """
        masks: list[torch.Tensor] = []
        for path_str in mask_paths:
            resolved = self._resolve_path(path_str)
            try:
                mask_pil = Image.open(resolved).convert("L")
                mask_np = np.array(mask_pil, dtype=np.float32)
                mask_np = (mask_np > 127).astype(np.float32)
                masks.append(torch.from_numpy(mask_np))
            except (FileNotFoundError, OSError):
                # Skip missing/corrupted mask files rather than crashing training
                continue
        return masks


    def _resolve_path(self, path: str) -> str:
        candidates = []
        raw = Path(path)
        if raw.is_absolute():
            candidates.append(raw)
        else:
            candidates.extend([
                Path(path),
                Path.cwd() / path,
            ])
            if self.vis_root:
                candidates.extend([
                    Path(self.vis_root) / path,
                    Path(self.vis_root) / raw.name,
                ])

        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return str(candidates[0]) if candidates else path

    @staticmethod
    def _boxes_to_tensor(boxes: list[dict]) -> torch.Tensor:
        coords = []
        for box in boxes:
            bbox = box.get("bbox")
            if bbox is None or len(bbox) != 4:
                continue
            coords.append([float(v) for v in bbox])
        if not coords:
            return torch.zeros((0, 4), dtype=torch.float32)
        return torch.tensor(coords, dtype=torch.float32)

    def _build_instruction_answer(self, info: dict, tasks: dict, K: int,
                                  box_scales: list[str], anatomy_regions: list[str | None]) -> tuple[str, str]:
        available_tasks = list(self.task_pool)
        # `identify` requires at least one valid bbox to query about; skip if K=0
        if "identify" in available_tasks and K <= 0:
            available_tasks = [t for t in available_tasks if t != "identify"]
        # `refer` is meaningless without a target either
        if "refer" in available_tasks and K <= 0:
            available_tasks = [t for t in available_tasks if t != "refer"]

        vqa_pairs = tasks.get("vqa") or []
        if vqa_pairs:
            available_tasks.append("vqa")
        task = random.choice(available_tasks) if available_tasks else "report"

        boxes_raw = tasks.get("boxes", []) or []

        if task == "report":
            prompt = "[report] Describe this ultrasound image in detail."
            answer = tasks.get("report") or self._fallback_report(info, K)
        elif task == "segmentation":
            prompt = "[segmentation] Segment all visible lesions or nodules."
            answer = self._segmentation_answer(K)
        elif task == "detection":
            prompt = "[detection] Locate all visible lesions or nodules."
            answer = self._detection_answer(K, box_scales)
        elif task == "refer":
            prompt = "[refer] Where is the lesion or nodule located?"
            answer = self._refer_answer(K, anatomy_regions)
        elif task == "identify":
            prompt, answer = self._identify_prompt_answer(
                info, boxes_raw, anatomy_regions
            )
        else:
            qa = random.choice(vqa_pairs)
            prompt = qa.get("question", "[vqa] Answer the question about this ultrasound image.")
            answer = qa.get("answer", "")
            if not prompt.startswith("[vqa]"):
                prompt = f"[vqa] {prompt}"

        processed_prompt = self.text_processor(prompt) if self.text_processor else prompt
        instruction = f"[INST] <Img><ImageHere></Img> {processed_prompt} [/INST]"
        return instruction, answer

    @staticmethod
    def _fallback_report(info: dict, K: int) -> str:
        anatomy = info.get("anatomy", "ultrasound")
        if K <= 0:
            return f"No lesion is visible on this {anatomy} ultrasound image."
        if K == 1:
            return f"One lesion is visible on this {anatomy} ultrasound image."
        return f"{K} lesions are visible on this {anatomy} ultrasound image."

    @staticmethod
    def _segmentation_answer(K: int) -> str:
        if K <= 0:
            return "No lesion is visible."
        tokens = " ".join(["<SEG>"] * K)
        lesion_word = "lesion" if K == 1 else "lesions"
        return f"{K} {lesion_word} should be segmented. {tokens}"

    @staticmethod
    def _detection_answer(K: int, box_scales: list[str]) -> str:
        if K <= 0:
            return "No lesion is visible."
        pieces = []
        for idx in range(K):
            scale = box_scales[idx] if idx < len(box_scales) else "M"
            token = {"S": "<BOX_S>", "M": "<BOX_M>", "L": "<BOX_L>"}.get(scale, "<BOX_M>")
            pieces.append(f"lesion {idx + 1} {token}")
        return "; ".join(pieces)

    @staticmethod
    def _refer_answer(K: int, anatomy_regions: list[str | None]) -> str:
        if K <= 0:
            return "No lesion is visible."
        region = next((r for r in anatomy_regions if r), None)
        if region:
            return f"The lesion is located in {region.replace('_', ' ')} <LOC>."
        return "The lesion location is indicated by <LOC>."

    # Prompt templates for the [identify] task — given a coordinate, name the object.
    # Mirrors IdentifyRSNADataset's instruction pool, adapted for ultrasound.
    _IDENTIFY_PROMPTS = (
        "[identify] what is at location {bbox}",
        "[identify] identify the object present at location {bbox}",
        "[identify] this region {bbox} contains",
        "[identify] describe the object inside {bbox}",
        "[identify] the structure within {bbox} is",
    )

    @staticmethod
    def _format_bbox_token(bbox: list[float]) -> str:
        """Render a bbox in the MiniGPT-v2 ``{<x1><y1><x2><y2>}`` token form."""
        if not bbox or len(bbox) != 4:
            return "{<0><0><0><0>}"
        x1, y1, x2, y2 = (int(round(float(v))) for v in bbox)
        return f"{{<{x1}><{y1}><{x2}><{y2}>}}"

    @classmethod
    def _identify_prompt_answer(cls, info: dict, boxes_raw: list[dict],
                                anatomy_regions: list[str | None]) -> tuple[str, str]:
        """Build an `[identify]` (location → label) supervision pair.

        The model is shown a single random bbox from the frame and must answer
        with the lesion class plus, when available, the parsed anatomy region.
        Falls back to a refer-style prompt if no usable bbox exists.
        """
        if not boxes_raw:
            # Should be filtered out upstream; defensive fallback.
            return ("[identify] what is the lesion in this image",
                    "No lesion is visible.")

        idx = random.randrange(len(boxes_raw))
        target = boxes_raw[idx]
        bbox = target.get("bbox") or []
        class_label = target.get("class") or "lesion"
        # Prefer this box's own anatomy_region; fall back to any frame-level one.
        region = target.get("anatomy_region")
        if not region:
            region = next((r for r in anatomy_regions if r), None)

        bbox_token = cls._format_bbox_token(bbox)
        prompt_tpl = random.choice(cls._IDENTIFY_PROMPTS)
        prompt = prompt_tpl.format(bbox=bbox_token)

        if region:
            answer = (f"<p>{class_label}</p> in {region.replace('_', ' ')} "
                      f"{bbox_token}")
        else:
            answer = f"<p>{class_label}</p> {bbox_token}"
        return prompt, answer

    def collater(self, samples: list[dict]) -> dict[str, Any]:
        # P2-15: Pad variable-length boxes to batch-max K for efficient batching
        all_boxes = [s["boxes"] for s in samples]
        max_k = max(b.shape[0] for b in all_boxes) if all_boxes else 0
        padded_boxes = []
        box_padding_mask = []
        for b in all_boxes:
            k = b.shape[0]
            if max_k > 0:
                if k < max_k:
                    pad = torch.zeros((max_k - k, 4), dtype=b.dtype)
                    b = torch.cat([b, pad], dim=0)
                mask = torch.zeros(max_k, dtype=torch.bool)
                mask[:k] = True
            else:
                b = torch.zeros((0, 4), dtype=torch.float32)
                mask = torch.zeros(0, dtype=torch.bool)
            padded_boxes.append(b)
            box_padding_mask.append(mask)

        return {
            "image": default_collate([sample["image"] for sample in samples]),
            "instruction_input": [sample["instruction_input"] for sample in samples],
            "answer": [sample["answer"] for sample in samples],
            "image_id": [sample["image_id"] for sample in samples],
            "modality": [sample["modality"] for sample in samples],
            "anatomy": [sample["anatomy"] for sample in samples],
            "boxes": torch.stack(padded_boxes) if max_k > 0 else all_boxes,
            "box_padding_mask": torch.stack(box_padding_mask) if max_k > 0 else box_padding_mask,
            "box_scales": [sample["box_scales"] for sample in samples],
            "anatomy_regions": [sample["anatomy_regions"] for sample in samples],
            "mask_paths": [sample["mask_paths"] for sample in samples],
            "masks": [sample["masks"] for sample in samples],
            "K": default_collate([sample["K"] for sample in samples]),
            "has_structured_supervision": default_collate([
                sample["has_structured_supervision"] for sample in samples
            ]),
            "raw_tasks": [sample["raw_tasks"] for sample in samples],
        }


class GroupBreastUSDataset(UnifiedUSDataset):
    def __init__(self, vis_processor, text_processor, vis_root: str, ann_path: str):
        super().__init__(vis_processor, text_processor, vis_root, ann_path, anatomy="breast")


class GroupThyroidUSDataset(UnifiedUSDataset):
    def __init__(self, vis_processor, text_processor, vis_root: str, ann_path: str):
        super().__init__(vis_processor, text_processor, vis_root, ann_path, anatomy="thyroid")


# ---------------------------------------------------------------------------
# Evaluation-time dataset
# ---------------------------------------------------------------------------
class evalGroupUSDataset(Dataset):
    """Deterministic per-task evaluation dataset for Group-Breast / Group-Thyroid US.

    Unlike :class:`UnifiedUSDataset` which randomly samples a task on every
    ``__getitem__`` call (good for training, bad for evaluation), this class
    materializes one (frame, task) pair per row, so a single epoch covers the
    full test set under a fixed task.

    Parameters
    ----------
    loaded_data : list[dict]
        Pre-loaded annotation list (the unified-schema JSON contents).
    vis_processor : callable
        Same processor used during training (image-side).
    root_path : str
        Directory under which ``image_path`` is resolved when relative.
    task : str
        One of ``{"report", "segmentation", "detection", "refer", "identify"}``.
    anatomy : str
        ``"breast"`` or ``"thyroid"`` (used only for prompt fall-backs).

    Returns per-item
    ----------------
    image, question, image_id, ground_truth
        ``ground_truth`` is task-specific and ready to be consumed by metrics:

        * ``report``       — gold report string
        * ``segmentation`` — list of resolved instance mask file paths
        * ``detection``    — list of [x1,y1,x2,y2] bboxes
        * ``refer``        — list of [x1,y1,x2,y2] bboxes (gold target boxes)
        * ``identify``     — gold class label string
    """

    _PROMPTS: dict = {
        "report":       "[report] Describe this ultrasound image in detail.",
        "segmentation": "[segmentation] Segment all visible lesions or nodules.",
        "detection":    "[detection] Locate all visible lesions or nodules.",
        "refer":        "[refer] Where is the lesion or nodule located?",
        # `identify` is filled per-sample with the bbox token.
    }

    def __init__(self, loaded_data: list, vis_processor, root_path: str,
                 task: str = "report", anatomy: str = "ultrasound"):
        assert task in {"report", "segmentation", "detection", "refer", "identify"}, \
            f"unsupported eval task: {task}"
        self.loaded_data = loaded_data
        self.vis_processor = vis_processor
        self.root_path = root_path
        self.task = task
        self.anatomy = anatomy

    def __len__(self):
        return len(self.loaded_data)

    def _resolve(self, path: str) -> str:
        p = Path(path)
        if p.is_absolute() and p.exists():
            return str(p)
        if self.root_path:
            cand = Path(self.root_path) / path
            if cand.exists():
                return str(cand)
            cand = Path(self.root_path) / p.name
            if cand.exists():
                return str(cand)
        return path

    def _load_image(self, image_path: str) -> torch.Tensor:
        resolved = self._resolve(image_path)
        grayscale_image = Image.open(resolved).convert("L")
        image = Image.new("RGB", grayscale_image.size)
        image.paste(grayscale_image)
        return self.vis_processor(image)

    def __getitem__(self, index):
        info = self.loaded_data[index]
        tasks = info.get("tasks", {}) or {}
        boxes_raw = tasks.get("boxes", []) or []

        image = self._load_image(info.get("image_path", ""))
        image_id = info.get("image_id", str(index))

        if self.task == "identify":
            # For identify we deterministically pick the FIRST bbox so that
            # repeated runs are reproducible.
            if not boxes_raw:
                # Skip-able edge case; return a trivial sample
                bbox_token = "{<0><0><0><0>}"
                gt = "no_lesion"
                question = f"[identify] what is at location {bbox_token}"
            else:
                target = boxes_raw[0]
                bbox = target.get("bbox") or [0, 0, 0, 0]
                gt = target.get("class") or "lesion"
                bbox_token = UnifiedUSDataset._format_bbox_token(bbox)
                question = f"[identify] what is at location {bbox_token}"
            return image, question, image_id, gt

        question = self._PROMPTS[self.task]

        if self.task == "report":
            gt = tasks.get("report") or ""
            return image, question, image_id, gt

        if self.task == "segmentation":
            mask_paths = [self._resolve(m.get("mask_path", ""))
                          for m in tasks.get("masks", []) if m.get("mask_path")]
            return image, question, image_id, mask_paths

        if self.task in ("detection", "refer"):
            bboxes = [b.get("bbox") for b in boxes_raw if b.get("bbox")]
            return image, question, image_id, bboxes

        # Should be unreachable due to assert in __init__
        raise RuntimeError(f"unsupported task: {self.task}")
