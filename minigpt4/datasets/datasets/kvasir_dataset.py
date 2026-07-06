"""Kvasir-SEG polyp dataset for OmniRad multi-task training and evaluation.

Supports 5 tasks (from mask → bbox + scale derivation):
  segmentation — emit K <SEG> tokens, one per polyp
  detection    — emit <BOX_S/M/L> tokens per polyp
  refer        — produce <LOC> token
  identify     — given a bbox, answer "polyp"
  vqa          — template QA pairs following Kvasir metadata.csv style

Consumes unified-schema JSON produced by ``tools/converters/kvasir.py``.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate


class KvasirDataset(Dataset):
    """Training dataset for Kvasir-SEG colonoscopy polyp images (mask + bbox).

    Randomly samples one task per ``__getitem__`` call.
    """

    def __init__(self, vis_processor, text_processor, vis_root: str, ann_path: str):
        self.vis_root = vis_root
        self.vis_processor = vis_processor
        self.text_processor = text_processor

        with open(ann_path, "r", encoding="utf-8") as f:
            ann = json.load(f)
        if isinstance(ann, dict):
            ann = ann.get("annotations", ann.get("data", []))
        self.ann = ann

        self.task_pool = ["segmentation", "detection", "refer", "identify"]

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

        available = list(self.task_pool)
        # identify / refer require K > 0
        if K <= 0:
            available = [t for t in available if t not in ("refer", "identify")]
        vqa_pairs = tasks.get("vqa") or []
        if vqa_pairs:
            available.append("vqa")
        task = random.choice(available) if available else "detection"

        instruction, answer = self._build_instruction(
            info, tasks, K, box_scales, anatomy_regions, vqa_pairs, task
        )

        return {
            "image": image,
            "instruction_input": instruction,
            "answer": answer,
            "image_id": info.get("image_id", str(index)),
            "modality": info.get("modality", "endoscopy"),
            "anatomy": info.get("anatomy", "colon"),
            "boxes": boxes,
            "box_scales": box_scales,
            "anatomy_regions": anatomy_regions,
            "mask_paths": mask_paths_raw,
            "masks": masks,
            "K": torch.tensor(K, dtype=torch.long),
            "has_structured_supervision": torch.tensor(True, dtype=torch.bool),
            "raw_tasks": tasks,
        }

    # ------------------------------------------------------------------
    # Image / mask loading
    # ------------------------------------------------------------------

    def _load_image(self, info: dict) -> torch.Tensor:
        path = self._resolve(info.get("image_path", ""))
        pil_img = Image.open(path).convert("RGB")
        return self.vis_processor(pil_img)

    def _load_masks(self, mask_paths: list[str]) -> list[torch.Tensor]:
        masks: list[torch.Tensor] = []
        for path_str in mask_paths:
            resolved = self._resolve(path_str)
            try:
                mask_pil = Image.open(resolved).convert("L")
                mask_np = np.array(mask_pil, dtype=np.float32)
                mask_np = (mask_np > 127).astype(np.float32)
                masks.append(torch.from_numpy(mask_np))
            except (FileNotFoundError, OSError):
                continue
        return masks

    def _resolve(self, path: str) -> str:
        raw = Path(path)
        if raw.is_absolute() and raw.exists():
            return str(raw)
        candidates = [Path(path), Path.cwd() / path]
        if self.vis_root:
            candidates.extend([
                Path(self.vis_root) / path,
                Path(self.vis_root) / raw.name,
            ])
        for c in candidates:
            if c.exists():
                return str(c)
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

    # ------------------------------------------------------------------
    # Prompt / answer construction
    # ------------------------------------------------------------------

    def _build_instruction(self, info, tasks, K, box_scales,
                           anatomy_regions, vqa_pairs, task) -> tuple[str, str]:
        boxes_raw = tasks.get("boxes", [])

        if task == "segmentation":
            prompt = "[segmentation] Segment all visible polyps."
            answer = self._segmentation_answer(K)
        elif task == "detection":
            prompt = "[detection] Locate all visible polyps."
            answer = self._detection_answer(K, box_scales)
        elif task == "refer":
            prompt = "[refer] Where is the polyp located?"
            answer = self._refer_answer(K, anatomy_regions)
        elif task == "identify":
            prompt, answer = self._identify_pair(info, boxes_raw, anatomy_regions)
        else:  # vqa
            qa = random.choice(vqa_pairs)
            prompt = qa.get("question", "[vqa] Describe this colonoscopy image.")
            answer = qa.get("answer", "")
            if not prompt.startswith("[vqa]"):
                prompt = f"[vqa] {prompt}"

        processed = self.text_processor(prompt) if self.text_processor else prompt
        instruction = f"[INST] <Img><ImageHere></Img> {processed} [/INST]"
        return instruction, answer

    # ------------------------------------------------------------------
    # Answer builders
    # ------------------------------------------------------------------

    @staticmethod
    def _segmentation_answer(K):
        if K <= 0:
            return "No polyp is visible."
        tokens = " ".join(["<SEG>"] * K)
        word = "polyp" if K == 1 else "polyps"
        return f"{K} {word} should be segmented. {tokens}"

    @staticmethod
    def _detection_answer(K, box_scales):
        if K <= 0:
            return "No polyp is visible."
        pieces = []
        for idx in range(K):
            scale = box_scales[idx] if idx < len(box_scales) else "M"
            token = {"S": "<BOX_S>", "M": "<BOX_M>", "L": "<BOX_L>"}.get(scale, "<BOX_M>")
            pieces.append(f"polyp {idx + 1} {token}")
        return "; ".join(pieces)

    @staticmethod
    def _refer_answer(K, anatomy_regions):
        if K <= 0:
            return "No polyp is visible."
        region = next((r for r in anatomy_regions if r), None)
        if region:
            return f"The polyp is located in {region.replace('_', ' ')} <LOC>."
        return "The polyp location is indicated by <LOC>."

    _IDENTIFY_PROMPTS = (
        "[identify] what is at location {bbox}",
        "[identify] identify the object present at location {bbox}",
        "[identify] this region {bbox} contains",
        "[identify] describe the object inside {bbox}",
    )

    @staticmethod
    def _format_bbox_token(bbox):
        if not bbox or len(bbox) != 4:
            return "{<0><0><0><0>}"
        x1, y1, x2, y2 = (int(round(float(v))) for v in bbox)
        return f"{{<{x1}><{y1}><{x2}><{y2}>}}"

    @classmethod
    def _identify_pair(cls, info, boxes_raw, anatomy_regions):
        if not boxes_raw:
            return ("[identify] what is the lesion in this image", "No polyp is visible.")
        idx = random.randrange(len(boxes_raw))
        target = boxes_raw[idx]
        bbox = target.get("bbox") or []
        class_label = target.get("class") or "polyp"
        region = target.get("anatomy_region") or next((r for r in anatomy_regions if r), None)
        bbox_token = cls._format_bbox_token(bbox)
        prompt_tpl = random.choice(cls._IDENTIFY_PROMPTS)
        prompt = prompt_tpl.format(bbox=bbox_token)
        if region:
            answer = f"<p>{class_label}</p> in {region.replace('_', ' ')} {bbox_token}"
        else:
            answer = f"<p>{class_label}</p> {bbox_token}"
        return prompt, answer

    # ------------------------------------------------------------------
    # Collation
    # ------------------------------------------------------------------

    def collater(self, samples: list[dict]) -> dict:
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
            "image": default_collate([s["image"] for s in samples]),
            "instruction_input": [s["instruction_input"] for s in samples],
            "answer": [s["answer"] for s in samples],
            "image_id": [s["image_id"] for s in samples],
            "modality": [s["modality"] for s in samples],
            "anatomy": [s["anatomy"] for s in samples],
            "boxes": torch.stack(padded_boxes) if max_k > 0 else all_boxes,
            "box_padding_mask": torch.stack(box_padding_mask) if max_k > 0 else box_padding_mask,
            "box_scales": [s["box_scales"] for s in samples],
            "anatomy_regions": [s["anatomy_regions"] for s in samples],
            "mask_paths": [s["mask_paths"] for s in samples],
            "masks": [s["masks"] for s in samples],
            "K": default_collate([s["K"] for s in samples]),
            "has_structured_supervision": default_collate([
                s["has_structured_supervision"] for s in samples
            ]),
            "raw_tasks": [s["raw_tasks"] for s in samples],
        }


# ---------------------------------------------------------------------------
# Evaluation-time dataset — deterministic per-task eval
# ---------------------------------------------------------------------------

class evalKvasirDataset(Dataset):
    """Deterministic per-task evaluation dataset for Kvasir-SEG.

    Supports 5 tasks: segmentation, detection, refer, identify, vqa.
    """

    _PROMPTS = {
        "detection":    "[detection] Locate all visible polyps.",
        "refer":        "[refer] Where is the polyp located?",
        "segmentation": "[segmentation] Segment all visible polyps.",
    }

    def __init__(self, loaded_data: list, vis_processor, root_path: str,
                 task: str = "segmentation"):
        assert task in {"segmentation", "detection", "refer", "identify", "vqa"}, \
            f"unsupported Kvasir eval task: {task}"
        self.loaded_data = loaded_data
        self.vis_processor = vis_processor
        self.root_path = root_path
        self.task = task

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
        return self.vis_processor(Image.open(resolved).convert("RGB"))

    def __getitem__(self, index):
        info = self.loaded_data[index]
        tasks = info.get("tasks", {}) or {}
        boxes_raw = tasks.get("boxes", []) or []
        image = self._load_image(info.get("image_path", ""))
        image_id = info.get("image_id", str(index))

        if self.task == "identify":
            if not boxes_raw:
                return image, "[identify] what is at location {<0><0><0><0>}", image_id, "no_polyp"
            target = boxes_raw[0]
            bbox = target.get("bbox") or [0, 0, 0, 0]
            gt = target.get("class") or "polyp"
            bbox_token = KvasirDataset._format_bbox_token(bbox)
            question = f"[identify] what is at location {bbox_token}"
            return image, question, image_id, gt

        if self.task == "vqa":
            vqa_pairs = tasks.get("vqa", [])
            # Deterministically pick the first VQA pair
            if vqa_pairs:
                qa = vqa_pairs[0]
                question = f"[vqa] {qa['question']}"
                gt = qa['answer']
            else:
                question = "[vqa] How many polyps are in the image?"
                gt = "0"
            return image, question, image_id, gt

        question = self._PROMPTS[self.task]

        if self.task == "segmentation":
            mask_paths = [self._resolve(m.get("mask_path", ""))
                          for m in tasks.get("masks", []) if m.get("mask_path")]
            return image, question, image_id, mask_paths

        if self.task in ("detection", "refer"):
            bboxes = [b.get("bbox") for b in boxes_raw if b.get("bbox")]
            return image, question, image_id, bboxes

        return image, question, image_id, None
