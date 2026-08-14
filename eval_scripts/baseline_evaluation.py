"""Baseline model evaluation script.

Runs any supported baseline model on the same test sets used by OmniRad,
producing ``summary.json`` files that ``compare_results.py`` can harvest into
comparison tables.

Supported baselines
-------------------
1. **minigpt_med** — Uses the existing MiniGPT-v2 architecture + MiniGPT-Med
   pretrained checkpoint.  This is the primary baseline since OmniRad inherits
   from MiniGPT-Med.

2. **minigpt_v2** — The original MiniGPT-v2 (stage-3, before medical FT).
   Uses the same architecture as minigpt_med but with the stage-3 checkpoint.

3. **chexagent** / **llava_med** / **med_flamingo** / **radfm** / **xraygpt** —
   Third-party medical VLMs from the ``models/`` directory.  They expose a
   ``BaseLM.generate(paths, prompt)`` interface (paths = list of image file
   paths, prompt = plain text).  This script bypasses the tensor-based
   DataLoader used by MiniGPT-v2 and calls each model directly on raw image
   paths, then writes predictions in the JSON layout expected by the shared
   metrics (``VQA_BERT_Sim`` / ``evaluate_report_generation``).

Usage
-----
    # MiniGPT-Med on all public datasets
    # CheXagent 报告生成
    python eval_scripts/baseline_evaluation.py \
    --model chexagent \
    --dataset indiana_cxr \
    --output-dir eval_results/chexagent

    # LLaVA-Med 报告 + VQA 全跑
    python eval_scripts/baseline_evaluation.py \
    --model llava_med \
    --dataset indiana_cxr,radvqa,slake_vqa \
    --output-dir eval_results/llava_med


Output
------
Each dataset's results are written under ``<output-dir>/<dataset>/``:
    - ``inference_results.json``  — raw model predictions
    - ``bert_sim.csv`` / ``bleu4.csv`` / ``rouge_l.csv`` / ``chexbert_f1.csv``
    - ``summary.json`` — aggregated metrics (consumed by compare_results.py)
"""

import sys
sys.path.append('.')

import os
import json
import argparse
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from minigpt4.common.config import Config
from minigpt4.common.eval_utils import prepare_texts, init_model, eval_parser, computeIoU
from minigpt4.conversation.conversation import CONV_VISION_minigptv2

from minigpt4.datasets.datasets.radvqa_dataset import evalRadVQADataset
from minigpt4.datasets.datasets.rsna_dataset import evalRSNADataset
from minigpt4.datasets.datasets.SLAKE_dataset import evalSLAKEDataset
from minigpt4.datasets.datasets.indiana_dataset import evalIndianaCXRDataset

from eval_scripts.clean_json import clean_report_json, clean_vqa_json, clean_detection_json
from eval_scripts.metrics import (
    evaluate_report_generation,
    VQA_BERT_Sim,
    average_iou,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
BASELM_MODELS = {"chexagent", "llava_med", "med_flamingo", "radfm", "xraygpt",
                  "biomedclip"}


def list_of_str(arg):
    return list(map(str, arg.split(',')))


def _decode_answer(ans):
    """OmniRad.generate returns dicts; MiniGPT-v2 returns strings."""
    if isinstance(ans, dict):
        return ans.get("text", "")
    return str(ans)


def _write_summary(output_dir: str, model_name: str, task_name: str,
                   metrics: dict, n_samples: int):
    """Write a summary.json consumed by compare_results.py."""
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


# ---------------------------------------------------------------------------
# Per-task evaluation functions
# ---------------------------------------------------------------------------
def eval_report_generation(model, vis_processor, conv_temp, cfg_block,
                           dataset_name, model_name, output_dir):
    """Evaluate report generation: BERT-Sim + BLEU-4 + ROUGE-L + CheXbert-F1."""
    eval_file_path = cfg_block["eval_file_path"]
    img_path = cfg_block["img_path"]
    batch_size = cfg_block.get("batch_size", 10)
    max_new_tokens = cfg_block.get("max_new_tokens", 300)

    with open(eval_file_path, "r") as f:
        data = json.load(f)

    eval_set = evalIndianaCXRDataset(data, vis_processor, img_path)
    loader = DataLoader(eval_set, batch_size=batch_size, shuffle=False)
    predictions = defaultdict(list)

    for images, questions, img_ids in tqdm(loader, desc=f"[{model_name}/{dataset_name}] report"):
        texts = prepare_texts(list(questions), conv_temp)
        answers = model.generate(images, texts,
                                max_new_tokens=max_new_tokens, do_sample=False)
        for ans, iid, q in zip(answers, img_ids, questions):
            predictions[iid].append(_decode_answer(ans))

    # Save raw predictions
    pred_path = os.path.join(output_dir, "inference_results.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(pred_path, "w") as f:
        json.dump(predictions, f)
    clean_report_json(pred_path, pred_path)

    # Run all metrics
    results = evaluate_report_generation(
        gt_pth=eval_file_path,
        pred_pth=pred_path,
        dataset_name=f"{model_name}_{dataset_name}",
        output_dir=output_dir,
    )

    _write_summary(output_dir, model_name, dataset_name, results, len(data))


def eval_vqa(model, vis_processor, conv_temp, cfg_block,
              dataset_name, model_name, output_dir):
    """Evaluate VQA: BERT-Sim."""
    eval_file_path = cfg_block["eval_file_path"]
    img_path = cfg_block["img_path"]
    batch_size = cfg_block.get("batch_size", 10)
    max_new_tokens = cfg_block.get("max_new_tokens", 300)

    with open(eval_file_path, "r") as f:
        data = json.load(f)

    eval_set = evalRadVQADataset(data, vis_processor, img_path)
    loader = DataLoader(eval_set, batch_size=batch_size, shuffle=False)
    predictions = defaultdict(list)

    for images, questions, img_ids in tqdm(loader, desc=f"[{model_name}/{dataset_name}] VQA"):
        texts = prepare_texts(list(questions), conv_temp)
        answers = model.generate(images, texts,
                                max_new_tokens=max_new_tokens, do_sample=False)
        for ans, iid, q in zip(answers, img_ids, questions):
            predictions[iid].append({
                "key": iid,
                "question": q.replace("[vqa]", "").strip(),
                "answer": _decode_answer(ans),
            })

    os.makedirs(output_dir, exist_ok=True)
    pred_path = os.path.join(output_dir, "inference_results.json")
    with open(pred_path, "w") as f:
        json.dump(predictions, f)
    clean_vqa_json(pred_path, pred_path)

    csv_path = os.path.join(output_dir, "vqa_bert_sim.csv")
    VQA_BERT_Sim(eval_file_path, pred_path, csv_path)

    # Parse average from CSV
    df = pd.read_csv(csv_path)
    avg_sim = df["BERT_score"].mean() if "BERT_score" in df.columns else 0.0
    print(f"  Average VQA BERT-Sim: {avg_sim:.4f}")

    _write_summary(output_dir, model_name, dataset_name,
                   {"bert_sim": avg_sim}, len(data))


def eval_detection(model, vis_processor, conv_temp, cfg_block,
                    dataset_name, model_name, output_dir):
    """Evaluate detection (RSNA): average IoU."""
    eval_file_path = cfg_block["eval_file_path"]
    img_path = cfg_block["img_path"]
    batch_size = cfg_block.get("batch_size", 10)
    max_new_tokens = cfg_block.get("max_new_tokens", 100)

    with open(eval_file_path, "r") as f:
        data = json.load(f)

    eval_set = evalRSNADataset(data, vis_processor, img_path)
    loader = DataLoader(eval_set, batch_size=batch_size, shuffle=False)
    predictions = defaultdict(list)

    for images, questions, img_ids in tqdm(loader, desc=f"[{model_name}/{dataset_name}] detection"):
        texts = prepare_texts(list(questions), conv_temp)
        answers = model.generate(images, texts,
                                max_new_tokens=max_new_tokens, do_sample=False)
        for ans, iid, q in zip(answers, img_ids, questions):
            predictions[iid].append(_decode_answer(ans))

    os.makedirs(output_dir, exist_ok=True)
    pred_path = os.path.join(output_dir, "inference_results.json")
    with open(pred_path, "w") as f:
        json.dump(predictions, f)

    csv_path = os.path.join(output_dir, "iou_results.csv")
    clean_detection_json(pred_path, pred_path)
    original_size = 1024  # RSNA images are 1024×1024
    avg_iou = average_iou(eval_file_path, pred_path, original_size, 100,
                          f"{model_name}_{dataset_name}", csv_path)

    _write_summary(output_dir, model_name, dataset_name,
                   {"iou": avg_iou}, len(data))


def eval_grounding(model, vis_processor, conv_temp, cfg_block,
                    dataset_name, model_name, output_dir):
    """Evaluate grounding (SLAKE): average IoU."""
    eval_file_path = cfg_block["eval_file_path"]
    img_path = cfg_block["img_path"]
    batch_size = cfg_block.get("batch_size", 10)
    max_new_tokens = cfg_block.get("max_new_tokens", 100)

    with open(eval_file_path, "r") as f:
        data = json.load(f)

    eval_set = evalSLAKEDataset(data, vis_processor, img_path)
    loader = DataLoader(eval_set, batch_size=batch_size, shuffle=False)
    predictions = defaultdict(list)

    for images, questions, img_ids in tqdm(loader, desc=f"[{model_name}/{dataset_name}] grounding"):
        texts = prepare_texts(list(questions), conv_temp)
        answers = model.generate(images, texts,
                                max_new_tokens=max_new_tokens, do_sample=False)
        for ans, iid, q in zip(answers, img_ids, questions):
            predictions[iid].append(_decode_answer(ans))

    os.makedirs(output_dir, exist_ok=True)
    pred_path = os.path.join(output_dir, "inference_results.json")
    with open(pred_path, "w") as f:
        json.dump(predictions, f)

    csv_path = os.path.join(output_dir, "iou_results.csv")
    clean_detection_json(pred_path, pred_path)
    avg_iou = average_iou(eval_file_path, pred_path, 100, 100,
                          f"{model_name}_{dataset_name}", csv_path)

    _write_summary(output_dir, model_name, dataset_name,
                   {"iou": avg_iou}, len(data))


# ---------------------------------------------------------------------------
# Task dispatch
# ---------------------------------------------------------------------------
TASK_DISPATCH = {
    # dataset_name → (eval_function, task_category)
    "indiana_cxr":   (eval_report_generation, "report_generation"),
    "radvqa":        (eval_vqa, "vqa"),
    "slake_vqa":     (eval_vqa, "vqa"),
    "rsna":          (eval_detection, "detection"),
    "SLAKE":         (eval_grounding, "grounding"),
}


# --------------------------------------------------------------------------- #
# BaseLM model integration (chexagent / llava_med / med_flamingo / radfm / xraygpt)
# --------------------------------------------------------------------------- #
_BASELM_IMPORT_MAP = {
    "chexagent":    ("models.chexagent",    "CheXagent"),
    "llava_med":    ("models.llavamed",     "LlavaMed"),
    "med_flamingo": ("models.medflamingo",  "MedFlamingo"),
    "radfm":        ("models.radfm",        "RadFM"),
    "xraygpt":      ("models.xraygpt",      "XrayGPT"),
    "biomedclip":   ("models.biomedclip",   "BiomedCLIP"),
}


def _init_baselm_model(model_name: str):
    """Lazy-import and instantiate a BaseLM model from the ``models/`` package."""
    if model_name not in _BASELM_IMPORT_MAP:
        raise ValueError(f"Unknown BaseLM model: {model_name}")
    import importlib
    module_path, class_name = _BASELM_IMPORT_MAP[model_name]
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    print(f"\n=> Loading {class_name} ({model_name})")
    return cls()


def _load_baselm_cfg():
    """Load the BaseLM benchmark YAML to obtain evaluation_datasets_cfg paths."""
    cfg_args = argparse.Namespace(
        cfg_path="eval_configs/baselm_benchmark_evaluation.yaml",
        options=None,
        gpu_id=0,
    )
    return Config(cfg_args)


# Default report-generation prompt — mirrors ``INDIANA_INSTRUCTION_POOL`` first
# entry so all baselines see the same instruction wording.
_BASELM_REPORT_PROMPT = "Describe this image in detail"

# VQA prompt prefix matching the OmniRad convention so the BERT-Sim metric
# strips it consistently via ``clean_vqa_json``.
_BASELM_VQA_PREFIX = "[vqa]"


def _eval_baselm_dataset(model, model_name: str, dataset_name: str,
                         cfg_block: dict, output_dir: str):
    """Dispatch a BaseLM model on a single dataset.

    Supports:
      - ``indiana_cxr`` → report generation (BERT-Sim + BLEU + ROUGE + CheXbert)
      - ``radvqa`` / ``slake_vqa`` → VQA (BERT-Sim)

    Other dataset/task combinations (detection, grounding) are not supported
    because the BaseLM interface only returns free-text — there is no dense
    head to decode boxes from.  Callers should pass only text-task datasets.
    """
    eval_file_path = cfg_block["eval_file_path"]
    img_path = cfg_block["img_path"]

    # BiomedCLIP is a contrastive model (no generation).  It can only do VQA
    # via zero-shot answer selection, not report generation.
    if model_name == "biomedclip" and dataset_name == "indiana_cxr":
        print(f"  [skip] BiomedCLIP has no generation capability; "
              f"report generation is not supported.")
        return

    with open(eval_file_path, "r") as f:
        data = json.load(f)

    if dataset_name == "indiana_cxr":
        _eval_baselm_report(model, model_name, data, img_path,
                            eval_file_path, output_dir)
    elif dataset_name in ("radvqa", "slake_vqa"):
        _eval_baselm_vqa(model, model_name, data, img_path,
                         dataset_name, eval_file_path, output_dir)
    else:
        print(f"  [skip] BaseLM models only support text tasks; "
              f"'{dataset_name}' requires dense heads.")


def _resolve_image_path(img_root: str, image_name: str) -> str:
    """Resolve an image name to an absolute path, handling missing extensions."""
    p = os.path.join(img_root, image_name)
    if os.path.exists(p):
        return p
    # Indiana images have a .png suffix that JSON ``image_id`` omits.
    for ext in (".png", ".jpg", ".jpeg"):
        cand = p + ext
        if os.path.exists(cand):
            return cand
    return p


def _eval_baselm_report(model, model_name, data, img_path, gt_pth, output_dir):
    """Run report generation on Indiana-style data and compute all metrics."""
    predictions = []
    for info in tqdm(data, desc=f"[{model_name}/indiana_cxr] report"):
        img_id = info["image_id"]
        image_path = _resolve_image_path(img_path, img_id)
        try:
            pred = model.generate([image_path], _BASELM_REPORT_PROMPT)
        except Exception as e:
            print(f"  [warn] generate failed for {img_id}: {e}")
            pred = ""
        predictions.append({"image_id": img_id, "caption": str(pred).strip()})

    pred_path = os.path.join(output_dir, "inference_results.json")
    with open(pred_path, "w") as f:
        json.dump(predictions, f)
    clean_report_json(pred_path, pred_path)

    results = evaluate_report_generation(
        gt_pth=gt_pth,
        pred_pth=pred_path,
        dataset_name=f"{model_name}_indiana_cxr",
        output_dir=output_dir,
    )
    _write_summary(output_dir, model_name, "indiana_cxr", results, len(data))


def _eval_baselm_vqa(model, model_name, data, img_path,
                     dataset_name, gt_pth, output_dir):
    """Run VQA on RadVQA / SLAKE-style data and compute BERT-Sim.

    Each entry in ``data`` has ``image_name``/``img_name``, ``question`` and
    ``answer``.  Predictions are grouped per image so multiple questions on the
    same image reuse the (already-loaded) model state without re-encoding the
    image — though each ``generate()`` call still processes the image once
    because BaseLM does not expose a separate encode step.
    """
    predictions = defaultdict(list)
    for info in tqdm(data, desc=f"[{model_name}/{dataset_name}] VQA"):
        # RadVQA uses ``image_name``, SLAKE uses ``img_name``
        img_name = info.get("image_name") or info.get("img_name") or ""
        image_path = _resolve_image_path(img_path, img_name)
        question = info.get("question", "")
        prompt = f"{_BASELM_VQA_PREFIX} {question}"
        try:
            pred = model.generate([image_path], prompt)
        except Exception as e:
            print(f"  [warn] generate failed for {img_name}: {e}")
            pred = ""
        predictions[img_name].append({
            "key": img_name,
            "question": question,
            "answer": str(pred).strip(),
        })

    pred_path = os.path.join(output_dir, "inference_results.json")
    with open(pred_path, "w") as f:
        json.dump(predictions, f)
    clean_vqa_json(pred_path, pred_path)

    csv_path = os.path.join(output_dir, "vqa_bert_sim.csv")
    VQA_BERT_Sim(gt_pth, pred_path, csv_path)

    df = pd.read_csv(csv_path)
    avg_sim = df["BERT_score"].mean() if "BERT_score" in df.columns else 0.0
    print(f"  Average VQA BERT-Sim: {avg_sim:.4f}")

    _write_summary(output_dir, model_name, dataset_name,
                   {"bert_sim": avg_sim}, len(data))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Baseline model evaluation")
    parser.add_argument("--model", type=str, required=True,
                        choices=["minigpt_med", "minigpt_v2", "llava"]
                                + sorted(BASELM_MODELS),
                        help="Which baseline model to evaluate.")
    parser.add_argument("--cfg-path", type=str, default=None,
                        help="Path to eval YAML config (for minigpt_med/minigpt_v2).")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Override checkpoint path.")
    parser.add_argument("--dataset", type=list_of_str, required=True,
                        help="Comma-separated dataset names to evaluate.")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Root output directory (e.g. eval_results/minigpt_med).")
    args = parser.parse_args()

    # --- Initialize model ---
    # For minigpt_med / minigpt_v2, we reuse the existing eval config system.
    if args.model in ("minigpt_med", "minigpt_v2"):
        if args.cfg_path is None:
            # Use sensible defaults
            if args.model == "minigpt_med":
                args.cfg_path = "eval_configs/minigptv2_benchmark_evaluation.yaml"
            else:
                args.cfg_path = "eval_configs/minigptv2_eval.yaml"

        # Build a Config from the YAML + CLI args
        # We need to merge --ckpt override if provided
        cfg_args = argparse.Namespace(
            cfg_path=args.cfg_path,
            options=None,
            gpu_id=0,
        )
        cfg = Config(cfg_args)

        # Override checkpoint if specified
        if args.ckpt:
            cfg.model_cfg.ckpt = args.ckpt

        model, vis_processor = init_model(cfg_args)
        model.eval()

        conv_temp = CONV_VISION_minigptv2.copy()
        conv_temp.system = ""
        save_path = args.output_dir

    elif args.model in BASELM_MODELS:
        # --- BaseLM models (chexagent / llava_med / med_flamingo / radfm / xraygpt) ---
        # They expose a ``generate(paths, prompt)`` interface and manage their own
        # image preprocessing internally, so we don't need vis_processor here.
        model = _init_baselm_model(args.model)
        vis_processor = None
        conv_temp = None
        save_path = args.output_dir

        # Build a stub cfg so dataset dispatch works. BaseLM eval reads paths
        # directly from JSON, so we use the benchmark config just for paths.
        cfg = _load_baselm_cfg()

    # --- Run evaluations ---
    os.makedirs(save_path, exist_ok=True)

    for dataset_name in args.dataset:
        print(f"\n{'='*60}")
        print(f"  Evaluating [{args.model}] on [{dataset_name}]")
        print(f"{'='*60}")

        if dataset_name not in cfg.evaluation_datasets_cfg:
            print(f"  [skip] dataset '{dataset_name}' not found in config")
            continue

        cfg_block = cfg.evaluation_datasets_cfg[dataset_name]
        output_dir = os.path.join(save_path, dataset_name)
        os.makedirs(output_dir, exist_ok=True)

        try:
            if args.model in BASELM_MODELS:
                # BaseLM models have their own image preprocessing and a
                # path-based generate() interface — bypass tensor DataLoader.
                _eval_baselm_dataset(
                    model=model,
                    model_name=args.model,
                    dataset_name=dataset_name,
                    cfg_block=cfg_block,
                    output_dir=output_dir,
                )
            else:
                eval_fn, task_category = TASK_DISPATCH[dataset_name]
                eval_fn(
                    model=model,
                    vis_processor=vis_processor,
                    conv_temp=conv_temp,
                    cfg_block=cfg_block,
                    dataset_name=dataset_name,
                    model_name=args.model,
                    output_dir=output_dir,
                )
        except Exception as e:
            print(f"  [error] evaluation failed: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"  All evaluations complete. Results in: {save_path}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
