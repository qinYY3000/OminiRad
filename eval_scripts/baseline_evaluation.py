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

3. **llava** (planned) — LLaVA-1.5 zero-shot.  Requires the ``llava`` package.
   Not yet implemented; see ``_eval_llava`` stub.

Usage
-----
    # MiniGPT-Med on all public datasets (including Kvasir multi-task)
    python eval_scripts/baseline_evaluation.py \
        --model minigpt_med \
        --cfg-path eval_configs/minigptv2_benchmark_evaluation.yaml \
        --dataset indiana_cxr,radvqa,slake_vqa,rsna,SLAKE,kvasir \
        --output-dir eval_results/minigpt_med

    # Single dataset
    python eval_scripts/baseline_evaluation.py \
        --model minigpt_med \
        --cfg-path eval_configs/minigptv2_benchmark_evaluation.yaml \
        --dataset indiana_cxr \
        --output-dir eval_results/minigpt_med

    # Kvasir multi-task evaluation only
    python eval_scripts/baseline_evaluation.py \
        --model minigpt_med \
        --cfg-path eval_configs/minigptv2_benchmark_evaluation.yaml \
        --dataset kvasir \
        --output-dir eval_results/minigpt_med

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
from minigpt4.datasets.datasets.kvasir_dataset import evalKvasirDataset

from eval_scripts.clean_json import clean_report_json, clean_vqa_json, clean_detection_json
from eval_scripts.metrics import (
    evaluate_report_generation,
    VQA_BERT_Sim,
    average_iou,
    average_dice,
    identify_accuracy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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
# Shared helpers for multi-task evaluation (US / Kvasir)
# ---------------------------------------------------------------------------
def _us_collate(batch):
    """Custom collate that keeps the variable-length / non-tensor ``ground_truth``
    field as a Python list (DataLoader's default would try to stack)."""
    images = torch.stack([b[0] for b in batch], dim=0)
    questions = [b[1] for b in batch]
    img_ids = [b[2] for b in batch]
    gts = [b[3] for b in batch]
    return images, questions, img_ids, gts


def _save_pred_masks(masks, out_dir: str, image_id: str,
                     threshold: float = 0.5) -> list:
    """Persist a list of (H, W) sigmoid-prob tensors as 0/255 PNG files."""
    if not masks:
        return []
    paths = []
    for k, m in enumerate(masks):
        try:
            arr = m.detach().cpu().numpy() if hasattr(m, "detach") else np.asarray(m)
        except Exception:
            continue
        binary = (arr >= threshold).astype("uint8") * 255
        out_path = os.path.join(out_dir, f"{image_id}_seg{k}.png")
        Image.fromarray(binary).save(out_path)
        paths.append(out_path)
    return paths


def _us_bbox_iou(gts, preds, csv_path, dataset_name):
    """Parse {<x1><y1><x2><y2>} tokens from predictions, IoU vs gold bboxes."""
    bbox_pat = re.compile(r"\{<(\d+)><(\d+)><(\d+)><(\d+)>\}")
    pred_index = {p["image_id"]: p["answer"] for p in preds}
    rows = []
    ious = []
    for gt in gts:
        iid = gt["image_id"]
        gt_boxes = gt.get("answer") or []
        pred_text = pred_index.get(iid, "") or ""
        pred_boxes = [list(map(int, m)) for m in bbox_pat.findall(pred_text)]
        if not gt_boxes or not pred_boxes:
            rows.append({"image_id": iid, "IoU": 0.0})
            ious.append(0.0)
            continue
        per_box = []
        for gb in gt_boxes:
            best = max((computeIoU(gb, pb) for pb in pred_boxes), default=0.0)
            per_box.append(best)
        frame_iou = sum(per_box) / len(per_box)
        rows.append({"image_id": iid, "IoU": frame_iou})
        ious.append(frame_iou)
    avg = sum(ious) / len(ious) if ious else 0.0
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Average IoU for dataset {dataset_name}: {avg:.4f}")
    return avg


# ---------------------------------------------------------------------------
# Kvasir — polyp multi-task evaluation
# ---------------------------------------------------------------------------
def eval_kvasir(model, vis_processor, conv_temp, cfg_block,
                dataset_name, model_name, output_dir):
    """Multi-task evaluation on Kvasir colonoscopy polyp test split.

    Runs segmentation, detection, refer, identify, vqa tasks.
    Note: segmentation requires mask generation (``return_masks=True``) which
    is only available in OmniRad — baseline models will skip it gracefully.
    """
    eval_file_path = cfg_block["eval_file_path"]
    img_path = cfg_block["img_path"]
    tasks_to_run = cfg_block.get("tasks", ["segmentation", "detection", "refer", "identify", "vqa"])
    default_bs = cfg_block.get("batch_size", 4)
    default_mnt = cfg_block.get("max_new_tokens", 120)

    with open(eval_file_path, "r") as f:
        ann = json.load(f)
    if isinstance(ann, dict):
        ann = ann.get("annotations", ann.get("data", []))

    summary = {}
    for task in tasks_to_run:
        print(f"\n========== [{model_name}/{dataset_name}] task = {task} ==========")

        # Baseline MiniGPT-v2 cannot generate masks — skip segmentation.
        if task == "segmentation" and model_name in ("minigpt_med", "minigpt_v2"):
            print("  [skip] segmentation requires mask decoder (OmniRad only for baselines).")
            continue

        task_cfg = (cfg_block.get("tasks_cfg") or {}).get(task, {}) or {}
        batch_size = task_cfg.get("batch_size", default_bs)
        max_new_tokens = task_cfg.get("max_new_tokens", default_mnt)

        eval_set = evalKvasirDataset(
            loaded_data=ann,
            vis_processor=vis_processor,
            root_path=img_path,
            task=task,
        )
        loader = DataLoader(eval_set, batch_size=batch_size, shuffle=False,
                            collate_fn=_us_collate)

        pred_mask_dir = os.path.join(output_dir, f"{task}_pred_masks")
        if task == "segmentation":
            os.makedirs(pred_mask_dir, exist_ok=True)

        predictions: list = []
        gts_compact: list = []

        for images, questions, img_ids, gts in tqdm(loader, desc=f"[{model_name}/{dataset_name}] {task}"):
            texts = prepare_texts(list(questions), conv_temp)
            gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": False}
            if task == "segmentation":
                gen_kwargs["return_masks"] = True
            outputs = model.generate(images, texts, **gen_kwargs)

            for out, iid, q, gt in zip(outputs, img_ids, questions, gts):
                ans_text = _decode_answer(out)
                # Also extract raw_text / masks when available (OmniRad dict output)
                raw_text = ans_text
                masks = []
                if isinstance(out, dict):
                    raw_text = out.get("raw_text", ans_text)
                    masks = out.get("masks", [])

                pred_record = {"image_id": iid, "answer": ans_text,
                               "raw": raw_text, "question": q}

                if task == "segmentation":
                    saved_paths = _save_pred_masks(masks, pred_mask_dir, iid)
                    pred_record["mask_paths"] = saved_paths
                    gt_record = {"image_id": iid,
                                 "tasks": {"masks": [{"mask_path": p}
                                                     for p in gt]}}
                else:
                    gt_record = {"image_id": iid, "answer": gt}

                predictions.append(pred_record)
                gts_compact.append(gt_record)

        pred_path = os.path.join(output_dir, f"{task}_inference_result.json")
        with open(pred_path, "w") as f:
            json.dump(predictions, f)
        gt_path = os.path.join(output_dir, f"{task}_gt.json")
        with open(gt_path, "w") as f:
            json.dump(gts_compact, f)

        csv_path = os.path.join(output_dir, f"{task}_metric.csv")

        if task in ("detection", "refer"):
            score = _us_bbox_iou(gts_compact, predictions, csv_path,
                                 dataset_name=f"{model_name}_{dataset_name}_{task}")
            summary[task] = score
            print(f"[{dataset_name}/{task}] avg-IoU = {score:.4f}")
        elif task == "segmentation":
            score = average_dice(gt_path, pred_path,
                                 dataset_name=f"{model_name}_{dataset_name}_{task}",
                                 csv_filename=csv_path)
            summary[task] = score
            print(f"[{dataset_name}/{task}] avg-Dice = {score:.4f}")
        elif task == "identify":
            score = identify_accuracy(gt_path, pred_path,
                                      f"{model_name}_{dataset_name}_{task}", csv_path)
            summary[task] = score
            print(f"[{dataset_name}/{task}] accuracy = {score:.4f}")
        elif task == "vqa":
            # Save GT and pred in format expected by VQA_BERT_Sim
            gt_vqa_list = []
            pred_dict = {}
            for p in predictions:
                iid = p["image_id"]
                q_text = p["question"].replace("[vqa]", "").strip()
                pred_dict.setdefault(iid, []).append({
                    "key": iid,
                    "question": q_text,
                    "answer": p["answer"],
                })
            kvasir_ann = json.load(open(eval_file_path, "r", encoding="utf-8"))
            if isinstance(kvasir_ann, dict):
                kvasir_ann = kvasir_ann.get("annotations", kvasir_ann.get("data", []))
            for entry in kvasir_ann:
                iid = entry.get("image_id", "")
                vqa_pairs = entry.get("tasks", {}).get("vqa", [])
                if vqa_pairs:
                    qa = vqa_pairs[0]
                    gt_vqa_list.append({
                        "image_name": iid,
                        "question": qa["question"],
                        "answer": qa["answer"],
                    })
            gt_vqa_path = os.path.join(output_dir, f"{task}_gt.json")
            with open(gt_vqa_path, "w") as f:
                json.dump(gt_vqa_list, f, ensure_ascii=False, indent=2)
            with open(pred_path, "w") as f:
                json.dump(pred_dict, f, ensure_ascii=False, indent=2)
            clean_pred_path = os.path.join(output_dir, f"{task}_cleaned_inference_result.json")
            clean_vqa_json(pred_path, clean_pred_path)
            bert_csv = os.path.join(output_dir, f"{task}_bert_sim.csv")
            score = VQA_BERT_Sim(gt_vqa_path, clean_pred_path, bert_csv)
            summary[task] = score
            print(f"[{dataset_name}/{task}] BERT-Sim = {score:.4f}")
        else:
            print(f"  [warn] unknown task '{task}' — skipped.")

    print(f"\n========== [{model_name}/{dataset_name}] summary ==========")
    metrics_for_summary = {}
    for k, v in summary.items():
        print(f"  {k:<14}: {v}")
        metrics_for_summary[k] = v

    _write_summary(output_dir, model_name, f"{dataset_name}_multi_task",
                   metrics_for_summary, len(ann))


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
    "kvasir":        (eval_kvasir, "multi-task"),
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Baseline model evaluation")
    parser.add_argument("--model", type=str, required=True,
                        choices=["minigpt_med", "minigpt_v2", "llava"],
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

    elif args.model == "llava":
        print("[llava] LLaVA evaluation not yet implemented.")
        print("  To add: install llava package, load model, implement _eval_llava().")
        return

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

        eval_fn, task_category = TASK_DISPATCH[dataset_name]
        try:
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
