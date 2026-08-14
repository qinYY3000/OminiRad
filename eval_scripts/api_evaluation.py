"""API-based external model evaluation for VQA & Report Generation.

Sends medical images + text prompts to an OpenAI-compatible API endpoint and
scores the responses with the same metric functions used by ``model_evaluation.py``.
Results are written as ``summary.json`` files consumable by ``compare_results.py``.

Supported models
----------------
- ``qwen3_vl_8b`` — Qwen3-VL-8B-Instruct via vLLM / cloud API
- ``gpt_4o``       — OpenAI GPT-4o
- ``generic``      — neutral prompt fallback for any VL model

Supported tasks
---------------
- Report generation: indiana_cxr, group_breast_us (report only)
- VQA:              radvqa, slake_vqa, kvasir (vqa)
- Identify:         kvasir (identify)

Usage
-----
    python eval_scripts/api_evaluation.py \\
        --model qwen3_vl_8b \\
        --api-base https://your-endpoint/v1 \\
        --api-key sk-xxx \\
        --cfg-path eval_configs/omnirad_evaluation.yaml \\
        --datasets indiana_cxr,radvqa,slake_vqa,kvasir \\
        --output-dir eval_results/qwen3_vl_8b

    # Dry-run to test connectivity
    python eval_scripts/api_evaluation.py \\
        --model qwen3_vl_8b \\
        --api-base http://localhost:8000/v1 \\
        --api-key not-needed \\
        --cfg-path eval_configs/omnirad_evaluation.yaml \\
        --datasets radvqa \\
        --max-samples 5 \\
        --dry-run
"""

import sys

sys.path.append(".")

import os
import re
import json
import argparse
import asyncio
import base64
import io
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from tqdm import tqdm
import yaml

# Heavy imports deferred to avoid loading model weights during --help / dry-run.
# Imported lazily in each eval function via _lazy_imports().


def _lazy_imports():
    """Import metric functions on-demand so ``--help`` works without model deps."""
    from eval_scripts.clean_json import clean_report_json, clean_vqa_json
    from eval_scripts.metrics import (
        evaluate_report_generation,
        VQA_BERT_Sim,
        identify_accuracy,
    )
    return clean_report_json, clean_vqa_json, evaluate_report_generation, VQA_BERT_Sim, identify_accuracy

# ============================================================================
# Prompt templates  (per-model adapters)
# ============================================================================

# Each entry returns a dict with optional "system" and the per-task
# user-prompt string.  The "vqa" template receives ``question`` as a
# format placeholder.  The "report" template receives nothing.

PROMPT_TEMPLATES = {
    "qwen3_vl_8b": {
        "report": {
            "system": (
                "You are a board-certified radiologist. "
                "Describe every clinically relevant finding in the medical image. "
                "Be thorough and precise."
            ),
            "user": (
                "Please write a detailed radiology report for this medical image. "
                "Include descriptions of all anatomical structures, any abnormalities, "
                "and an impression section."
            ),
        },
        "vqa": {
            "system": (
                "You are a medical expert. Answer the question based solely on "
                "what you see in the image. Be concise."
            ),
            "user": "{question}",
        },
        "vqa_kvasir": {
            "system": (
                "You are an expert endoscopist. Answer the question based on "
                "what you see in the colonoscopy image. Be concise and accurate."
            ),
            "user": "{question}",
        },
        "identify": {
            "system": (
                "You are a medical expert. Identify what is inside the marked "
                "bounding box region of this medical image."
            ),
            "user": (
                "The region marked by the bounding box {bbox} contains what? "
                "Answer with the class name only (e.g. polyp, lesion, nodule)."
            ),
        },
    },
    "gpt_4o": {
        "report": {
            "system": (
                "You are a board-certified radiologist. Generate a complete "
                "radiology report from the provided image."
            ),
            "user": (
                "Write a detailed radiology report for this medical image. "
                "Cover findings, impression, and recommendations."
            ),
        },
        "vqa": {
            "system": (
                "You are a medical expert. Answer concisely based on the image."
            ),
            "user": "{question}",
        },
        "vqa_kvasir": {
            "system": (
                "You are an expert endoscopist. Answer the question based on "
                "the colonoscopy image."
            ),
            "user": "{question}",
        },
        "identify": {
            "system": (
                "You are a medical expert. Identify what is inside the marked "
                "bounding box region."
            ),
            "user": (
                "What is inside the region {bbox}? Answer with one word."
            ),
        },
    },
    "generic": {
        "report": {
            "system": None,
            "user": (
                "Describe this medical image in detail. Include all relevant "
                "findings, observations, and an impression."
            ),
        },
        "vqa": {
            "system": None,
            "user": "{question}",
        },
        "vqa_kvasir": {
            "system": None,
            "user": "{question}",
        },
        "identify": {
            "system": None,
            "user": (
                "The region {bbox} in this medical image contains what? "
                "Answer with the class name only."
            ),
        },
    },
}

# ============================================================================
# Image encoding
# ============================================================================

MAX_IMAGE_PX = 2048
JPEG_QUALITY = 85


def encode_image_base64(image_path: str, max_px: int = MAX_IMAGE_PX) -> str:
    """Load image, optionally resize, and return a ``data:image/jpeg;base64,...`` URI."""
    pil = Image.open(image_path).convert("RGB")
    w, h = pil.size
    if max(w, h) > max_px:
        scale = max_px / max(w, h)
        pil = pil.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=JPEG_QUALITY)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _resolve_image(cfg_img_path: str, image_id_or_name: str) -> str:
    """Resolve an image file given a root dir and an image identifier.

    Tries common extensions (in dataset subdirs when nested).
    """
    root = Path(cfg_img_path)
    # Sometimes the id already contains a relative path; sometimes not.
    candidates = [
        root / image_id_or_name,
        root / f"{image_id_or_name}.jpg",
        root / f"{image_id_or_name}.png",
        root / f"{image_id_or_name}.jpeg",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    # Try glob inside root (1 level) for VQA-RAD ``synpic*`` style
    for ext in (".jpg", ".png", ".jpeg"):
        matches = list(root.glob(f"*{image_id_or_name}*{ext}"))
        if not matches:
            matches = list(root.glob(f"**/*{image_id_or_name}*{ext}"))
        if matches:
            return str(matches[0])
    # Last resort: assume it's a direct path join
    return str(root / image_id_or_name)


# ============================================================================
# API client
# ============================================================================

API_TIMEOUT = 120
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0


def _build_messages(template: dict, task: str, **fmt_kwargs):
    """Build an OpenAI-style messages list from a template dict."""
    msgs = []
    sys_msg = template.get("system")
    if sys_msg:
        msgs.append({"role": "system", "content": sys_msg})
    user_text = template["user"].format(**fmt_kwargs)
    msgs.append({"role": "user", "content": user_text})
    return msgs


async def _call_api(
    client,
    messages: list,
    base64_uri: str,
    model_name: str,
    max_tokens: int,
    sem: asyncio.Semaphore,
) -> str:
    """Send one image + messages to the API, retry on transient errors."""
    # Build the multimodal user-message content
    content = [
        {"type": "image_url", "image_url": {"url": base64_uri}},
        {"type": "text", "text": messages[-1]["content"]},
    ]
    # If there is a system message, keep it separate
    payload = []
    if len(messages) > 1 and messages[0]["role"] == "system":
        payload.append({"role": "system", "content": messages[0]["content"]})
    payload.append({"role": "user", "content": content})

    for attempt in range(1, MAX_RETRIES + 1):
        async with sem:
            try:
                resp = await client.chat.completions.create(
                    model=model_name,
                    messages=payload,
                    max_tokens=max_tokens,
                    temperature=0.0,
                    timeout=API_TIMEOUT,
                )
                return resp.choices[0].message.content or ""
            except Exception as exc:
                if attempt == MAX_RETRIES:
                    print(f"\n  [API error] attempt {attempt}/{MAX_RETRIES}: {exc}")
                    return ""
                wait = RETRY_BACKOFF ** attempt
                print(f"\n  [API retry {attempt}] {exc} — waiting {wait:.0f}s")
                await asyncio.sleep(wait)
    return ""


# ============================================================================
# Dataset loaders
# ============================================================================


def load_indiana_data(cfg_block: dict) -> list:
    """Return list of (image_path, image_id, ground_truth_caption)."""
    with open(cfg_block["eval_file_path"], "r", encoding="utf-8") as f:
        data = json.load(f)
    img_root = cfg_block["img_path"]
    samples = []
    for item in data:
        iid = item.get("image_id", "")
        caption = item.get("caption") or ""
        # Indiana images: stored as <img_root>/<image_id>.png
        img_p = _resolve_image(img_root, iid)
        samples.append((img_p, iid, caption))
    return samples


def load_vqa_data(cfg_block: dict) -> list:
    """Return list of (image_path, image_name, question, ground_truth_answer).

    Handles both RadVQA (``image_name`` key) and SLAKE-VQA (``img_name`` key).
    """
    with open(cfg_block["eval_file_path"], "r", encoding="utf-8") as f:
        data = json.load(f)
    img_root = cfg_block["img_path"]
    samples = []
    for item in data:
        # Auto-detect key
        img_name = item.get("image_name") or item.get("img_name") or ""
        question = item.get("question", "").strip()
        answer = item.get("answer", "").strip()
        if not img_name or not question:
            continue
        img_p = _resolve_image(img_root, img_name)
        samples.append((img_p, img_name, question, answer))
    return samples


def load_group_us_report_data(cfg_block: dict) -> list:
    """Return list of (image_path, image_id, ground_truth_report)."""
    with open(cfg_block["eval_file_path"], "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("annotations", data.get("data", []))
    img_root = cfg_block["img_path"]
    samples = []
    for item in data:
        iid = item.get("image_id", "")
        report = (item.get("tasks", {}) or {}).get("report") or ""
        img_path = item.get("image_path", "")
        resolved = _resolve_image(img_root, img_path) if img_path else ""
        if not resolved:
            resolved = _resolve_image(img_root, iid)
        if resolved:
            samples.append((resolved, iid, report))
    return samples


def load_kvasir_vqa_data(cfg_block: dict) -> list:
    """Return list of (image_path, image_id, question, ground_truth_answer)."""
    with open(cfg_block["eval_file_path"], "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("annotations", data.get("data", []))
    img_root = cfg_block["img_path"]
    samples = []
    for item in data:
        iid = item.get("image_id", "")
        vqa_pairs = (item.get("tasks", {}) or {}).get("vqa", [])
        img_path = item.get("image_path", "")
        resolved = _resolve_image(img_root, img_path) if img_path else _resolve_image(img_root, iid)
        if not resolved or not vqa_pairs:
            continue
        qa = vqa_pairs[0]
        samples.append((resolved, iid, qa.get("question", ""), qa.get("answer", "")))
    return samples


def load_kvasir_identify_data(cfg_block: dict) -> list:
    """Return list of (image_path, image_id, bbox_token, ground_truth_label)."""
    with open(cfg_block["eval_file_path"], "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("annotations", data.get("data", []))
    img_root = cfg_block["img_path"]
    samples = []
    for item in data:
        iid = item.get("image_id", "")
        boxes = (item.get("tasks", {}) or {}).get("boxes", [])
        if not boxes:
            continue
        box = boxes[0]
        bbox = box.get("bbox") or [0, 0, 0, 0]
        gt_label = box.get("class") or "polyp"
        bbox_token = f"{{<{int(bbox[0])}><{int(bbox[1])}><{int(bbox[2])}><{int(bbox[3])}>}}"
        img_path = item.get("image_path", "")
        resolved = _resolve_image(img_root, img_path) if img_path else _resolve_image(img_root, iid)
        if resolved:
            samples.append((resolved, iid, bbox_token, gt_label))
    return samples


# ============================================================================
# Evaluation loops
# ============================================================================


def _pick_template(model: str, task: str) -> dict:
    """Return the best matching template dict for *model* × *task*."""
    templates = PROMPT_TEMPLATES.get(model, PROMPT_TEMPLATES["generic"])
    return templates.get(task, PROMPT_TEMPLATES["generic"].get(task, {}))


async def eval_report_generation_api(
    client,
    model: str,
    cfg_block: dict,
    output_dir: str,
    sem: asyncio.Semaphore,
    max_samples: int = 0,
    dry_run: bool = False,
) -> dict:
    """Evaluate report generation on a dataset via API."""
    dataset_name = cfg_block.get("_dataset_name", "report")
    max_tokens = cfg_block.get("max_new_tokens", 512)
    samples = load_indiana_data(cfg_block)
    if max_samples and max_samples < len(samples):
        samples = samples[:max_samples]

    template = _pick_template(model, "report")

    predictions: dict = {}
    async def _one(sample):
        img_p, iid, gt = sample
        b64 = encode_image_base64(img_p)
        msgs = _build_messages(template, "report")
        ans = await _call_api(client, msgs, b64, model, max_tokens, sem)
        return iid, ans

    # --- Prepare all tasks ---
    tasks = [_one(s) for s in samples]
    for fut in tqdm(
        asyncio.as_completed(tasks),
        total=len(tasks),
        desc=f"[API/{model}/{dataset_name}] report",
    ):
        iid, ans = await fut
        predictions.setdefault(iid, []).append(ans)

    if dry_run:
        print(f"  [dry-run] collected {len(predictions)} responses")
        return {}

    clean_report_json, _, evaluate_report_generation, _, _ = _lazy_imports()

    os.makedirs(output_dir, exist_ok=True)
    pred_path = os.path.join(output_dir, "inference_results.json")
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f)
    clean_report_json(pred_path, pred_path)

    results = evaluate_report_generation(
        gt_pth=cfg_block["eval_file_path"],
        pred_pth=pred_path,
        dataset_name=f"{model}_{dataset_name}",
        output_dir=output_dir,
    )
    _write_summary(output_dir, model, dataset_name, results, len(samples))
    return results


async def eval_vqa_api(
    client,
    model: str,
    cfg_block: dict,
    output_dir: str,
    sem: asyncio.Semaphore,
    max_samples: int = 0,
    dry_run: bool = False,
) -> dict:
    """Evaluate VQA on radvqa / slake_vqa via API."""
    dataset_name = cfg_block.get("_dataset_name", "vqa")
    max_tokens = cfg_block.get("max_new_tokens", 128)
    samples = load_vqa_data(cfg_block)
    if max_samples and max_samples < len(samples):
        samples = samples[:max_samples]

    template = _pick_template(model, "vqa")
    predictions: dict = {}

    async def _one(sample):
        img_p, img_name, question, gt = sample
        b64 = encode_image_base64(img_p)
        msgs = _build_messages(template, "vqa", question=question)
        ans = await _call_api(client, msgs, b64, model, max_tokens, sem)
        return img_name, question, ans

    tasks = [_one(s) for s in samples]
    for fut in tqdm(
        asyncio.as_completed(tasks),
        total=len(tasks),
        desc=f"[API/{model}/{dataset_name}] VQA",
    ):
        img_name, question, ans = await fut
        predictions.setdefault(img_name, []).append({
            "key": img_name,
            "question": question,
            "answer": ans,
        })

    if dry_run:
        print(f"  [dry-run] collected {sum(len(v) for v in predictions.values())} responses")
        return {}

    _, clean_vqa_json, _, VQA_BERT_Sim, _ = _lazy_imports()

    os.makedirs(output_dir, exist_ok=True)
    pred_path = os.path.join(output_dir, "inference_results.json")
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f)

    # Clean & score
    clean_pred_path = os.path.join(output_dir, "cleaned_inference_results.json")
    clean_vqa_json(pred_path, clean_pred_path)
    csv_path = os.path.join(output_dir, "vqa_bert_sim.csv")
    avg_sim = VQA_BERT_Sim(cfg_block["eval_file_path"], clean_pred_path, csv_path)

    metrics = {"bert_sim": avg_sim}
    _write_summary(output_dir, model, dataset_name, metrics, len(samples))
    return metrics


async def eval_us_report_api(
    client,
    model: str,
    cfg_block: dict,
    output_dir: str,
    sem: asyncio.Semaphore,
    max_samples: int = 0,
    dry_run: bool = False,
) -> dict:
    """Evaluate report generation on group_breast_us via API."""
    dataset_name = cfg_block.get("_dataset_name", "group_breast_us_report")
    max_tokens = cfg_block.get("max_new_tokens", 512)
    samples = load_group_us_report_data(cfg_block)
    if max_samples and max_samples < len(samples):
        samples = samples[:max_samples]

    template = _pick_template(model, "report")
    predictions: list = []
    gts: list = []

    async def _one(sample):
        img_p, iid, gt = sample
        b64 = encode_image_base64(img_p)
        msgs = _build_messages(template, "report")
        ans = await _call_api(client, msgs, b64, model, max_tokens, sem)
        return iid, ans, gt

    tasks = [_one(s) for s in samples]
    for fut in tqdm(
        asyncio.as_completed(tasks),
        total=len(tasks),
        desc=f"[API/{model}/{dataset_name}] report",
    ):
        iid, ans, gt = await fut
        predictions.append({"image_id": iid, "answer": ans})
        gts.append({"image_id": iid, "answer": gt})

    if dry_run:
        print(f"  [dry-run] collected {len(predictions)} responses")
        return {}

    _, _, evaluate_report_generation, _, _ = _lazy_imports()

    os.makedirs(output_dir, exist_ok=True)
    pred_path = os.path.join(output_dir, "inference_results.json")
    gt_path = os.path.join(output_dir, "gt.json")
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f)
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(gts, f)

    results = evaluate_report_generation(
        gt_pth=gt_path,
        pred_pth=pred_path,
        dataset_name=f"{model}_{dataset_name}",
        output_dir=output_dir,
    )
    _write_summary(output_dir, model, dataset_name, results, len(samples))
    return results


async def eval_kvasir_vqa_api(
    client,
    model: str,
    cfg_block: dict,
    output_dir: str,
    sem: asyncio.Semaphore,
    max_samples: int = 0,
    dry_run: bool = False,
) -> dict:
    """Evaluate VQA on kvasir via API."""
    dataset_name = cfg_block.get("_dataset_name", "kvasir_vqa")
    max_tokens = cfg_block.get("max_new_tokens", 120)
    if "tasks_cfg" in cfg_block:
        vqa_cfg = (cfg_block.get("tasks_cfg") or {}).get("vqa", {}) or {}
        max_tokens = vqa_cfg.get("max_new_tokens", max_tokens)
    samples = load_kvasir_vqa_data(cfg_block)
    if max_samples and max_samples < len(samples):
        samples = samples[:max_samples]

    template = _pick_template(model, "vqa_kvasir")
    predictions: dict = {}
    gt_list: list = []

    async def _one(sample):
        img_p, iid, question, gt = sample
        b64 = encode_image_base64(img_p)
        msgs = _build_messages(template, "vqa_kvasir", question=question)
        ans = await _call_api(client, msgs, b64, model, max_tokens, sem)
        return iid, question, gt, ans

    tasks = [_one(s) for s in samples]
    for fut in tqdm(
        asyncio.as_completed(tasks),
        total=len(tasks),
        desc=f"[API/{model}/{dataset_name}] VQA",
    ):
        iid, question, gt, ans = await fut
        predictions.setdefault(iid, []).append({
            "key": iid,
            "question": question,
            "answer": ans,
        })
        gt_list.append({
            "image_name": iid,
            "question": question,
            "answer": gt,
        })

    if dry_run:
        print(f"  [dry-run] collected {sum(len(v) for v in predictions.values())} responses")
        return {}

    _, clean_vqa_json, _, VQA_BERT_Sim, _ = _lazy_imports()

    os.makedirs(output_dir, exist_ok=True)
    pred_path = os.path.join(output_dir, "inference_results.json")
    gt_path = os.path.join(output_dir, "gt.json")
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f)
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(gt_list, f)

    clean_pred_path = os.path.join(output_dir, "cleaned_inference_results.json")
    clean_vqa_json(pred_path, clean_pred_path)
    csv_path = os.path.join(output_dir, "vqa_bert_sim.csv")
    avg_sim = VQA_BERT_Sim(gt_path, clean_pred_path, csv_path)

    metrics = {"bert_sim": avg_sim}
    _write_summary(output_dir, model, dataset_name, metrics, len(samples))
    return metrics


async def eval_kvasir_identify_api(
    client,
    model: str,
    cfg_block: dict,
    output_dir: str,
    sem: asyncio.Semaphore,
    max_samples: int = 0,
    dry_run: bool = False,
) -> dict:
    """Evaluate identify on kvasir via API."""
    dataset_name = cfg_block.get("_dataset_name", "kvasir_identify")
    max_tokens = cfg_block.get("max_new_tokens", 60)
    if "tasks_cfg" in cfg_block:
        id_cfg = (cfg_block.get("tasks_cfg") or {}).get("identify", {}) or {}
        max_tokens = id_cfg.get("max_new_tokens", max_tokens)
    samples = load_kvasir_identify_data(cfg_block)
    if max_samples and max_samples < len(samples):
        samples = samples[:max_samples]

    template = _pick_template(model, "identify")
    predictions: list = []
    gts: list = []

    async def _one(sample):
        img_p, iid, bbox_token, gt_label = sample
        b64 = encode_image_base64(img_p)
        msgs = _build_messages(template, "identify", bbox=bbox_token)
        ans = await _call_api(client, msgs, b64, model, max_tokens, sem)
        return iid, ans, gt_label

    tasks = [_one(s) for s in samples]
    for fut in tqdm(
        asyncio.as_completed(tasks),
        total=len(tasks),
        desc=f"[API/{model}/{dataset_name}] identify",
    ):
        iid, ans, gt_label = await fut
        predictions.append({"image_id": iid, "answer": ans})
        gts.append({"image_id": iid, "answer": gt_label})

    if dry_run:
        print(f"  [dry-run] collected {len(predictions)} responses")
        return {}

    _, _, _, _, identify_accuracy = _lazy_imports()

    os.makedirs(output_dir, exist_ok=True)
    pred_path = os.path.join(output_dir, "inference_results.json")
    gt_path = os.path.join(output_dir, "gt.json")
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f)
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(gts, f)

    csv_path = os.path.join(output_dir, "identify_accuracy.csv")
    acc = identify_accuracy(gt_path, pred_path, f"{model}_{dataset_name}", csv_path)

    metrics = {"identify_accuracy": acc}
    _write_summary(output_dir, model, dataset_name, metrics, len(samples))
    return metrics


# ============================================================================
# Summary output  (compatible with compare_results.py)
# ============================================================================


def _write_summary(output_dir: str, model_name: str, task_name: str,
                   metrics: dict, n_samples: int):
    """Write ``summary.json`` consumable by ``compare_results.py``."""
    summary = {
        "model": model_name,
        "task": task_name,
        "metrics": metrics,
        "n_samples": n_samples,
        "timestamp": datetime.now().isoformat(),
    }
    path = os.path.join(output_dir, "summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  → summary written to {path}")


# ============================================================================
# Dispatch table
# ============================================================================

# Map dataset → (loader_fn, eval_fn, dataset_name_override)
# loader_fn:   loads samples (used to resolve images / count)
# eval_fn:     async coroutine that runs the API eval
# ds_name:     name used in summary.json "task" field
REGISTRY = {
    "indiana_cxr": {
        "eval": eval_report_generation_api,
        "ds_name": "indiana_cxr",
    },
    "radvqa": {
        "eval": eval_vqa_api,
        "ds_name": "radvqa",
    },
    "slake_vqa": {
        "eval": eval_vqa_api,
        "ds_name": "slake_vqa",
    },
    "group_breast_us": {
        "eval": eval_us_report_api,
        "ds_name": "group_breast_us_report",
    },
    "kvasir_vqa": {
        "eval": eval_kvasir_vqa_api,
        "ds_name": "kvasir_vqa",
    },
    "kvasir_identify": {
        "eval": eval_kvasir_identify_api,
        "ds_name": "kvasir_identify",
    },
}


# ============================================================================
# Main
# ============================================================================


def list_of_str(arg):
    return list(map(str, arg.split(",")))


async def main_async(args):
    # Build API client
    import openai

    client = openai.AsyncOpenAI(
        base_url=args.api_base,
        api_key=args.api_key or "not-needed",
    )
    sem = asyncio.Semaphore(args.concurrency)

    # Load YAML config
    with open(args.cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    eval_cfg = cfg.get("evaluation_datasets", {})

    output_root = args.output_dir
    os.makedirs(output_root, exist_ok=True)

    for ds in args.datasets:
        print(f"\n{'=' * 60}")
        print(f"  [{args.model}] → {ds}")
        print(f"{'=' * 60}")

        if ds not in eval_cfg and ds not in REGISTRY:
            print(f"  [skip] unknown dataset '{ds}' — not in config or registry")
            continue

        registry_entry = REGISTRY.get(ds)
        if registry_entry is None:
            print(f"  [skip] dataset '{ds}' has no API eval handler in REGISTRY")
            continue

        cfg_block = dict(eval_cfg.get(ds, {})) if ds in eval_cfg else {}
        cfg_block["_dataset_name"] = registry_entry["ds_name"]

        output_dir = os.path.join(output_root, registry_entry["ds_name"])
        eval_fn = registry_entry["eval"]

        try:
            metrics = await eval_fn(
                client=client,
                model=args.model,
                cfg_block=cfg_block,
                output_dir=output_dir,
                sem=sem,
                max_samples=args.max_samples,
                dry_run=args.dry_run,
            )
            print(f"  metrics: {json.dumps(metrics, indent=2) if metrics else '(dry-run)'}")
        except Exception as exc:
            print(f"  [error] evaluation failed: {exc}")
            import traceback
            traceback.print_exc()

    await client.close()

    print(f"\n{'=' * 60}")
    print(f"  Done. Results in: {output_root}/")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="API-based external model evaluation for medical VQA & report generation"
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Model identifier (qwen3_vl_8b, gpt_4o, generic, ...).")
    parser.add_argument("--api-base", type=str, required=True,
                        help="OpenAI-compatible API base URL (e.g. http://localhost:8000/v1).")
    parser.add_argument("--api-key", type=str, default="",
                        help="API key. Required for cloud APIs; optional for local vLLM.")
    parser.add_argument("--cfg-path", type=str, required=True,
                        help="Path to YAML eval config (e.g. eval_configs/omnirad_evaluation.yaml).")
    parser.add_argument("--datasets", type=list_of_str, required=True,
                        help="Comma-separated dataset names: indiana_cxr,radvqa,slake_vqa,kvasir_vqa,kvasir_identify,group_breast_us")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Root output directory (e.g. eval_results/qwen3_vl_8b).")
    parser.add_argument("--concurrency", type=int, default=8,
                        help="Max concurrent API requests (default: 8).")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Limit to first N samples per dataset (0 = all).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Fetch responses but skip metric computation (for connectivity test).")
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
