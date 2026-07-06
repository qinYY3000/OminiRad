"""OmniRad multi-dataset evaluator.

Example usage (single GPU)::

    torchrun --master-port 8888 --nproc_per_node 1 \\
        eval_scripts/model_evaluation.py \\
        --cfg-path eval_configs/omnirad_evaluation.yaml \\
        --dataset indiana_cxr,radvqa,slake_vqa,rsna,SLAKE,group_breast_us,kvasir
"""

import sys
sys.path.append('.')
import os
import re
import json
import argparse
from collections import defaultdict
import random
import numpy as np
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
from minigpt4.datasets.datasets.unified_us_dataset import evalGroupUSDataset
from minigpt4.datasets.datasets.kvasir_dataset import evalKvasirDataset
from minigpt4.datasets.datasets.indiana_dataset import evalIndianaCXRDataset
# NLST eval is disabled at the YAML level (raw images unavailable in this run).
# Re-enable by uncommenting the relevant block in eval_configs/omnirad_evaluation.yaml
# *and* importing eval_NLST_Dataset inside process_nlst_dataset() (see below).

from eval_scripts.clean_json import clean_report_json, clean_vqa_json, clean_detection_json
from eval_scripts.metrics import (
    report_bert_sim, VQA_BERT_Sim, average_iou,
    average_dice, identify_accuracy,
    evaluate_report_generation,
)


def list_of_str(arg):
    return list(map(str, arg.split(',')))


parser = eval_parser()
parser.add_argument("--dataset", type=list_of_str, help="dataset to evaluate")
args = parser.parse_args()
cfg = Config(args)


model, vis_processor = init_model(args)
model.eval()
CONV_VISION = CONV_VISION_minigptv2
conv_temp = CONV_VISION.copy()
conv_temp.system = ""
save_path = cfg.run_cfg.save_path


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _decode_answer(ans):
    """OmniRad.generate now returns a dict per sample. Fall back gracefully
    when the underlying model is the legacy text-only one (str output)."""
    if isinstance(ans, dict):
        return ans.get("text", "")
    return ans


# ---------------------------------------------------------------------------
# Indiana University Open-i — report generation
# ---------------------------------------------------------------------------
def process_indiana_dataset():
    eval_file_path = cfg.evaluation_datasets_cfg[dataset]["eval_file_path"]
    img_path = cfg.evaluation_datasets_cfg[dataset]["img_path"]
    batch_size = cfg.evaluation_datasets_cfg[dataset]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg[dataset]["max_new_tokens"]

    with open(eval_file_path, "r") as f:
        indiana = json.load(f)

    data = evalIndianaCXRDataset(indiana, vis_processor, img_path)
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    minigpt4_predict = defaultdict(list)

    for images, questions, img_ids in tqdm(eval_dataloader):
        texts = prepare_texts(questions, conv_temp)
        answers = model.generate(images, texts,
                                 max_new_tokens=max_new_tokens, do_sample=False)
        for answer, img_id, question in zip(answers, img_ids, questions):
            minigpt4_predict[img_id].append(_decode_answer(answer))

    file_save_path = os.path.join(save_path, "Indiana_inference_results.json")
    with open(file_save_path, "w") as f:
        json.dump(minigpt4_predict, f)
    clean_report_json(file_save_path, file_save_path)

    # Run all report-generation metrics: BERT-Sim + BLEU-4 + ROUGE-L + CheXbert-F1
    evaluate_report_generation(
        gt_pth=eval_file_path,
        pred_pth=file_save_path,
        dataset_name="indiana_cxr",
        output_dir=save_path,
    )


# ---------------------------------------------------------------------------
# VQA-RAD / SLAKE VQA — visual question answering
# ---------------------------------------------------------------------------
def process_vqa_dataset():
    eval_file_path = cfg.evaluation_datasets_cfg[dataset]["eval_file_path"]
    img_path = cfg.evaluation_datasets_cfg[dataset]["img_path"]
    batch_size = cfg.evaluation_datasets_cfg[dataset]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg[dataset]["max_new_tokens"]

    with open(eval_file_path, "r") as f:
        radVQA = json.load(f)

    data = evalRadVQADataset(radVQA, vis_processor, img_path)
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    minigpt4_predict = defaultdict(list)

    for images, questions, img_ids in tqdm(eval_dataloader):
        texts = prepare_texts(questions, conv_temp)
        answers = model.generate(images, texts,
                                 max_new_tokens=max_new_tokens, do_sample=False)
        for answer, img_id, question in zip(answers, img_ids, questions):
            minigpt4_predict[img_id].append({
                "key": img_ids,
                "question": question.replace("[vqa]", "").strip(),
                "answer": _decode_answer(answer),
            })

    file_save_path = os.path.join(save_path, "radVQA_inference_results.json")
    output_csv_path = os.path.join(save_path, "vqa_bert_similarity_scores.csv")

    with open(file_save_path, "w") as f:
        json.dump(minigpt4_predict, f)

    clean_vqa_json(file_save_path, file_save_path)
    VQA_BERT_Sim(eval_file_path, file_save_path, output_csv_path)


# ---------------------------------------------------------------------------
# RSNA — chest X-ray pneumonia detection (zero-shot)
# ---------------------------------------------------------------------------
def process_rsna_dataset():
    eval_file_path = cfg.evaluation_datasets_cfg[dataset]["eval_file_path"]
    img_path = cfg.evaluation_datasets_cfg[dataset]["img_path"]
    batch_size = cfg.evaluation_datasets_cfg[dataset]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg[dataset]["max_new_tokens"]

    with open(eval_file_path, "r") as f:
        rsna = json.load(f)

    data = evalRSNADataset(rsna, vis_processor, img_path)
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    minigpt4_predict = defaultdict(list)

    for images, questions, img_ids in tqdm(eval_dataloader):
        texts = prepare_texts(questions, conv_temp)
        answers = model.generate(images, texts,
                                 max_new_tokens=max_new_tokens, do_sample=False)
        for answer, img_id, question in zip(answers, img_ids, questions):
            minigpt4_predict[img_id].append(_decode_answer(answer))

    file_save_path = os.path.join(save_path, "RSNA_inference_result.json")
    with open(file_save_path, "w") as f:
        json.dump(minigpt4_predict, f)

    csv_pth = os.path.join(save_path, "RSNA_IoU_results.csv")
    clean_detection_json(file_save_path, file_save_path)
    average_iou(eval_file_path, file_save_path, 1024, 100, "rsna", csv_pth)


# ---------------------------------------------------------------------------
# SLAKE — grounded caption (Grounding task)
# ---------------------------------------------------------------------------
def process_SLAKE_dataset():
    eval_file_path = cfg.evaluation_datasets_cfg[dataset]["eval_file_path"]
    img_path = cfg.evaluation_datasets_cfg[dataset]["img_path"]
    batch_size = cfg.evaluation_datasets_cfg[dataset]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg[dataset]["max_new_tokens"]

    with open(eval_file_path, "r") as f:
        slake = json.load(f)

    data = evalSLAKEDataset(slake, vis_processor, img_path)
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    minigpt4_predict = defaultdict(list)

    for images, questions, img_ids in tqdm(eval_dataloader):
        texts = prepare_texts(questions, conv_temp)
        answers = model.generate(images, texts,
                                 max_new_tokens=max_new_tokens, do_sample=False)
        for answer, img_id, question in zip(answers, img_ids, questions):
            minigpt4_predict[img_id].append(_decode_answer(answer))

    file_save_path = os.path.join(save_path, "SLAKE_inference_result.json")
    with open(file_save_path, "w") as f:
        json.dump(minigpt4_predict, f)

    csv_pth = os.path.join(save_path, "SLAKE_IoU_results.csv")
    clean_detection_json(file_save_path, file_save_path)
    average_iou(eval_file_path, file_save_path, 100, 100, "SLAKE", csv_pth)


# ----------------------------------------------------------------------------
# Group-Breast US / Kvasir — multi-task evaluation
# ----------------------------------------------------------------------------
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


def process_group_us_dataset():
    """Run all 5 tasks (report / segmentation / detection / refer / identify)
    on a Group-Breast US test split.

    Per-dataset YAML must contain:

        eval_file_path : path to the unified-schema test JSON
        img_path       : root directory used for path resolution
        anatomy        : "breast"  (drives prompt fall-backs)
        tasks          : list of task names to run (default: all five)
        batch_size, max_new_tokens (per-task overrides under ``tasks_cfg`` allowed)
    """
    cfg_block = cfg.evaluation_datasets_cfg[dataset]
    eval_file_path = cfg_block["eval_file_path"]
    img_path = cfg_block["img_path"]
    anatomy = cfg_block.get("anatomy", "ultrasound")
    tasks_to_run = cfg_block.get(
        "tasks",
        ["report", "segmentation", "detection", "refer", "identify"],
    )
    default_bs = cfg_block.get("batch_size", 4)
    default_mnt = cfg_block.get("max_new_tokens", 200)

    with open(eval_file_path, "r") as f:
        ann = json.load(f)
    if isinstance(ann, dict):
        ann = ann.get("annotations", ann.get("data", []))

    summary = {}
    for task in tasks_to_run:
        print(f"\n========== [{dataset}] task = {task} ==========")
        task_cfg = (cfg_block.get("tasks_cfg") or {}).get(task, {}) or {}
        batch_size = task_cfg.get("batch_size", default_bs)
        max_new_tokens = task_cfg.get("max_new_tokens", default_mnt)

        eval_set = evalGroupUSDataset(
            loaded_data=ann,
            vis_processor=vis_processor,
            root_path=img_path,
            task=task,
            anatomy=anatomy,
        )
        loader = DataLoader(eval_set, batch_size=batch_size, shuffle=False,
                            collate_fn=_us_collate)

        predictions: list = []
        gts_compact: list = []
        pred_mask_dir = os.path.join(save_path, f"{dataset}_{task}_pred_masks")
        if task == "segmentation":
            os.makedirs(pred_mask_dir, exist_ok=True)

        for images, questions, img_ids, gts in tqdm(loader):
            texts = prepare_texts(list(questions), conv_temp)
            gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": False}
            if task == "segmentation":
                gen_kwargs["return_masks"] = True
            outputs = model.generate(images, texts, **gen_kwargs)

            for out, iid, q, gt in zip(outputs, img_ids, questions, gts):
                if isinstance(out, dict):
                    ans_text = out.get("text", "")
                    raw_text = out.get("raw_text", ans_text)
                    masks = out.get("masks", [])
                else:
                    ans_text = str(out)
                    raw_text = ans_text
                    masks = []

                pred_record = {
                    "image_id": iid,
                    "answer": ans_text,
                    "raw": raw_text,
                    "question": q,
                }

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

        pred_path = os.path.join(save_path,
                                 f"{dataset}_{task}_inference_result.json")
        with open(pred_path, "w") as f:
            json.dump(predictions, f)
        gt_path = os.path.join(save_path, f"{dataset}_{task}_gt.json")
        with open(gt_path, "w") as f:
            json.dump(gts_compact, f)

        csv_path = os.path.join(save_path, f"{dataset}_{task}_metric.csv")

        if task == "report":
            # Run BERT-Sim + BLEU-4 + ROUGE-L + CheXbert-F1 for US reports.
            # Note: CheXbert labels are chest-X-ray-specific, so for ultrasound
            # the CheXbert-F1 may be less meaningful — we still report it for
            # consistency and leave interpretation to the user.
            report_metrics = evaluate_report_generation(
                gt_pth=gt_path,
                pred_pth=pred_path,
                dataset_name=f"{dataset}_{task}",
                output_dir=save_path,
            )
            summary[task] = report_metrics
        elif task in ("detection", "refer"):
            score = _us_bbox_iou(gts_compact, predictions, csv_path,
                                 dataset_name=f"{dataset}_{task}")
            summary[task] = score
            print(f"[{dataset}/{task}] avg-IoU = {score:.4f}")
        elif task == "segmentation":
            score = average_dice(gt_path, pred_path,
                                 dataset_name=f"{dataset}_{task}",
                                 csv_filename=csv_path)
            summary[task] = score
            print(f"[{dataset}/{task}] avg-Dice = {score:.4f}")
        elif task == "identify":
            score = identify_accuracy(gt_path, pred_path,
                                      f"{dataset}_{task}", csv_path)
            summary[task] = score
        else:
            print(f"  [warn] unknown task '{task}' — skipped.")

    print(f"\n========== [{dataset}] summary ==========")
    for k, v in summary.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for mk, mv in v.items():
                print(f"    {mk:<20}: {mv:.4f}" if isinstance(mv, float) else f"    {mk:<20}: {mv}")
        else:
            print(f"  {k:<14}: {v}")


def _us_report_bert(gts, preds, csv_path):
    """BERT-Sim between gold reports and model predictions, ultrasound flavor."""
    from eval_scripts.metrics import compute_bert_similarity
    import pandas as _pd

    pred_index = {p["image_id"]: p["answer"] for p in preds}
    rows = []
    total = 0.0
    n = 0
    for gt in gts:
        iid = gt["image_id"]
        gt_text = gt.get("answer") or ""
        pred_text = pred_index.get(iid, "")
        if not gt_text or not pred_text:
            continue
        s = compute_bert_similarity(pred_text, gt_text)
        rows.append({"image_id": iid, "BERT_score": s})
        total += s
        n += 1
    avg = total / n if n else 0.0
    _pd.DataFrame(rows).to_csv(csv_path, index=False)
    return avg


def _us_bbox_iou(gts, preds, csv_path, dataset_name):
    """Parse {<x1><y1><x2><y2>} tokens from predictions, IoU vs gold bboxes."""
    import pandas as _pd

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
    _pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Average IoU for dataset {dataset_name}: {avg:.4f}")
    return avg


# ---------------------------------------------------------------------------
# Kvasir — polyp bbox-only evaluation (detection / refer / identify)
# ---------------------------------------------------------------------------
def process_kvasir_dataset():
    """Run multi-task evaluation on Kvasir colonoscopy polyp test split.

    Supported tasks: segmentation, detection, refer, identify.
    """
    cfg_block = cfg.evaluation_datasets_cfg[dataset]
    eval_file_path = cfg_block["eval_file_path"]
    img_path = cfg_block["img_path"]
    tasks_to_run = cfg_block.get("tasks", ["detection", "refer", "identify"])
    default_bs = cfg_block.get("batch_size", 4)
    default_mnt = cfg_block.get("max_new_tokens", 120)

    with open(eval_file_path, "r") as f:
        ann = json.load(f)
    if isinstance(ann, dict):
        ann = ann.get("annotations", ann.get("data", []))

    summary = {}
    for task in tasks_to_run:
        print(f"\n========== [{dataset}] task = {task} ==========")
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

        predictions: list = []
        gts_compact: list = []

        for images, questions, img_ids, gts in tqdm(loader):
            texts = prepare_texts(list(questions), conv_temp)
            gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": False}
            outputs = model.generate(images, texts, **gen_kwargs)

            for out, iid, q, gt in zip(outputs, img_ids, questions, gts):
                ans_text = out.get("text", "") if isinstance(out, dict) else str(out)
                pred_record = {"image_id": iid, "answer": ans_text, "question": q}
                predictions.append(pred_record)
                gts_compact.append({"image_id": iid, "answer": gt})

        pred_path = os.path.join(save_path, f"{dataset}_{task}_inference_result.json")
        with open(pred_path, "w") as f:
            json.dump(predictions, f)
        gt_path = os.path.join(save_path, f"{dataset}_{task}_gt.json")
        with open(gt_path, "w") as f:
            json.dump(gts_compact, f)

        csv_path = os.path.join(save_path, f"{dataset}_{task}_metric.csv")

        if task in ("detection", "refer"):
            score = _us_bbox_iou(gts_compact, predictions, csv_path,
                                 dataset_name=f"{dataset}_{task}")
            summary[task] = score
            print(f"[{dataset}/{task}] avg-IoU = {score:.4f}")
        elif task == "identify":
            score = identify_accuracy(gt_path, pred_path,
                                      f"{dataset}_{task}", csv_path)
            summary[task] = score
            print(f"[{dataset}/{task}] accuracy = {score:.4f}")
        else:
            print(f"  [warn] unknown task '{task}' — skipped.")

    print(f"\n========== [{dataset}] summary ==========")
    for k, v in summary.items():
        print(f"  {k:<14}: {v}")


############################################################################
# Dispatch
############################################################################
for dataset in args.dataset:
    if dataset == 'indiana_cxr':
        process_indiana_dataset()

    elif dataset == 'radvqa':
        process_vqa_dataset()

    elif dataset == 'rsna':
        process_rsna_dataset()

    elif dataset == 'SLAKE':
        process_SLAKE_dataset()

    elif dataset == 'group_breast_us':
        process_group_us_dataset()

    elif dataset == 'kvasir':
        process_kvasir_dataset()

    else:
        print(f"Dataset '{dataset}' is not supported.")
