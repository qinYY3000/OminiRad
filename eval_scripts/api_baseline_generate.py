#!/usr/bin/env python3
"""Call OpenAI-compatible VLM APIs for medical VQA / Report datasets.
Resumes from checkpoint, saves incrementally, handles rate limits gracefully.

Usage:
    python eval_scripts/api_baseline_generate.py --dataset kvasir --task vqa
    python eval_scripts/api_baseline_generate.py --dataset indiana --task report
"""

from __future__ import annotations
import argparse, base64, json, logging, os, re, sys, time
from io import BytesIO
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ============================================================
# PROMPT REGISTRY
# ============================================================
PROMPT_REGISTRY: dict[str, dict] = {
    "kvasir": {
        "vqa": {
            "system_prompt": (
                "You are a medical AI analyzing colonoscopy images. "
                "Answer each question with ONLY the answer — no explanation.\n"
                "- 'How many polyps/findings?' -> number: 0,1,2...\n"
                "- 'Is there a finding?' -> yes/no\n"
                "- 'Are there abnormalities?' -> polyp (or none)\n"
                "- 'Have all polyps been removed?' -> yes/no\n"
                "- 'What type of polyp?' -> polyp (not subtype)\n"
                "- 'What type of procedure?' -> colonoscopy\n"
                "- 'Is this finding easy to detect?' -> yes/no\n"
                "- 'What is the size of the polyp?' -> polyp 1: small/medium/large\n"
                "- 'Where is the abnormality?' -> upper-center/middle-center etc.\n"
                "- 'How many instruments?' -> 0,1,2...\n"
                "- 'Are there instruments?' -> none"
            ),
            "user_prefix": "[vqa]", "max_tokens": 16, "temperature": 0.0,
        },
    },
    "slake": {
        "vqa": {
            "system_prompt": (
                "You are a radiologist. Answer concisely — 1-5 words.\n"
                "Modality: 'X-Ray','CT' or 'MRI'. Body part: 'Chest','Abdomen','Head' etc.\n"
                "Yes/no: 'Yes'/'No'. Organ: 'Lung','Liver','Heart'.\n"
                "Abnormality: 'Left Lung, Lower Right'. Disease: 'Mass','Atelectasis','None'.\n"
                "Do NOT explain."
            ),
            "user_prefix": "[vqa]", "max_tokens": 32, "temperature": 0.0,
        },
        "grounding": {
            "system_prompt": "Radiologist: grounded description. Format: <p>struct</p> {<x1><y1><x2><y2>}, coords 0-100.",
            "user_prefix": "[grounding]",
            "user_template": "[grounding] please describe this image in details",
            "max_tokens": 256, "temperature": 0.0,
        },
    },
    "radvqa": {
        "vqa": {
            "system_prompt": (
                "You are a radiologist. Answer concisely — 1-5 words.\n"
                "Yes/no:'yes'/'no'. Abnormal: name or 'normal'. Organ: organ name.\nDo NOT explain."
            ),
            "user_prefix": "[vqa]", "max_tokens": 32, "temperature": 0.0,
        },
    },
    "indiana": {
        "report": {
            "system_prompt": (
                "You are a radiologist. Write a chest X-ray report in the "
                "standard Indiana University style: concise factual sentences "
                "describing what you see. Do NOT use section headers like "
                "'Findings:' — just state observations directly. Cover: heart "
                "size, lungs (opacities, effusions, nodules, masses), "
                "mediastinum, pleura, bones, tubes. End with 'Impression: ' "
                "followed by a one-sentence summary. Under 150 words. No IDs.\n"
                "Example: 'Heart size is normal. Lungs are clear bilaterally "
                "without consolidation or effusion. No pneumothorax. "
                "Impression: No acute cardiopulmonary abnormality.'"
            ),
            "user_prefix": "[report]",
            "user_template": "[report] Describe this chest X-ray in detail.",
            "max_tokens": 300, "temperature": 0.0,
        },
    },
    "group_breast_us": {
        "report": {
            "system_prompt": (
                "You are a radiologist writing a breast ultrasound report. "
                "Follow the standard structure with four paragraphs:\n"
                "1) Breast structure: 'The structure of both breasts is clear, "
                "with [uneven/heterogeneous] glandular echoes...'\n"
                "2) Nodule description (be precise): location (clock position, "
                "e.g. '10-11 o'clock direction'), distance from nipple (mm), "
                "size (length×width mm), orientation (parallel to the skin), "
                "shape (oval/elliptical), margins (smooth and well-defined), "
                "internal echogenicity (anechoic for cysts / hypoechoic for "
                "solid nodules), calcifications (present/absent), posterior "
                "echo (enhancement / no significant change), surrounding "
                "glands (not distorted), ducts (not dilated), overlying skin "
                "(no thickening/retraction).\n"
                "3) Axillary nodes: 'No significantly enlarged lymph nodes "
                "are observed in the bilateral axillary drainage areas.'\n"
                "4) Conclusion with BI-RADS: anechoic nodule → category 2; "
                "hypoechoic nodule → category 3. State it exactly as "
                "'...nodule, BI-RADS category X.'\n"
                "Write in clinical prose, under 200 words, no patient IDs."
            ),
            "user_prefix": "[report]",
            "user_template": "[report] Write a standard breast ultrasound report "
                            "following the structure: breast structure, nodule "
                            "details (clock position, distance, size, orientation, "
                            "shape, margins, echogenicity, calcifications, posterior "
                            "echo), axillary nodes, and BI-RADS conclusion.",
            "max_tokens": 300, "temperature": 0.0,
        },
    },
}

DATASET_DEFAULTS = {
    "kvasir":          {"image_dir":"data/kvasir/imgs",        "ann_file":"data/annotations/kvasir_test.json",      "output":"results/kvasir"},
    "slake":           {"image_dir":"data/slake/imgs",         "ann_file":"data/annotations/VQA_test_SLAKE.json",   "output":"results/slake"},
    "radvqa":          {"image_dir":"data/radvqa/imgs",        "ann_file":"data/annotations/vqa_test.json",          "output":"results/radvqa"},
    "indiana":         {"image_dir":"data/Chest X/images",     "ann_file":"data/annotations/indiana_test.json",     "output":"results/indiana"},
    "group_breast_us": {"image_dir":"data/group_breast",       "ann_file":"data/annotations/group_breast_test.json", "output":"results/group_breast_us"},
}

# ============================================================
# Arg parse
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description="API baseline generation")
    p.add_argument("--dataset", type=str, default="kvasir", choices=list(PROMPT_REGISTRY.keys()))
    p.add_argument("--task", type=str, default="vqa", choices=["vqa","report","grounding"])
    p.add_argument("--image-dir", type=str, default=None)
    p.add_argument("--ann-file", type=str, default=None)
    p.add_argument("--api-base-url", type=str, default="https://api-inference.modelscope.cn/v1")
    p.add_argument("--api-key", type=str, default="ms-xxxxxx")
    p.add_argument("--model-name", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--frequency-penalty", type=float, default=0.0)
    p.add_argument("--presence-penalty", type=float, default=0.0)
    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--retry-delay", type=float, default=10.0)
    p.add_argument("--request-delay", type=float, default=20.0)
    p.add_argument("--batch-questions", action="store_true", default=True)
    return p.parse_args()

def resolve_defaults(args):
    d = DATASET_DEFAULTS.get(args.dataset, {})
    if args.image_dir is None: args.image_dir = d.get("image_dir", "data/images")
    if args.ann_file is None:  args.ann_file  = d.get("ann_file", "data/test.json")
    if args.output is None:
        ms = args.model_name.split("/")[-1].lower()
        args.output = f"results/{args.dataset}/{args.task}_{ms}.json"
    return args

def get_prompt_config(dataset, task):
    ds = PROMPT_REGISTRY.get(dataset)
    if ds is None: sys.exit(f"Unknown dataset: {dataset}")
    tc = ds.get(task)
    if tc is None: sys.exit(f"No task '{task}' for '{dataset}'. Available: {list(ds.keys())}")
    return tc

# ============================================================
# Image helpers
# ============================================================
def encode_image_base64(path, max_size: int = 2048):
    p = Path(path)
    if not p.exists(): return None
    m = {".jpg":"jpeg",".jpeg":"jpeg",".png":"png",".webp":"webp",".bmp":"bmp"}
    # Resize large images to fit API limits (Qwen-VL: 2048x2048 max)
    from PIL import Image
    img = Image.open(p)
    w, h = img.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        new_size = (int(w * ratio), int(h * ratio))
        logging.info("Resizing %s: %dx%d → %dx%d", p.name, w, h, new_size[0], new_size[1])
        img = img.resize(new_size, Image.LANCZOS)
    buf = BytesIO()
    fmt = "JPEG" if p.suffix.lower() in (".jpg",".jpeg") else "PNG"
    img.save(buf, format=fmt)
    return f"data:image/{m.get(p.suffix.lower(),'jpeg')};base64,{base64.b64encode(buf.getvalue()).decode()}"

def resolve_image_path(path, img_dir):
    p = Path(path)
    for c in [p, Path(img_dir)/path, Path(img_dir)/p.name]:
        if c.exists(): return str(c)
    return str(p)

def build_user_prompt(question, cfg, info):
    if "user_template" in cfg: return cfg["user_template"]
    q = (question or "").strip()
    pf = cfg.get("user_prefix", "[vqa]")
    if not q.startswith(pf): q = f"{pf} {q}"
    return q

# ============================================================
# API call with progressive retry / rate-limit backoff
# ============================================================
def call_api(client, model, uri, prompt, system_prompt=None,
             max_tokens=32, temperature=0.0, top_p=1.0,
             frequency_penalty=0.0, presence_penalty=0.0):
    msgs = []
    if system_prompt: msgs.append({"role":"system","content":system_prompt})
    msgs.append({"role":"user","content":[{"type":"text","text":prompt},
                 {"type":"image_url","image_url":{"url":uri}}]})
    r = client.chat.completions.create(model=model, messages=msgs, max_tokens=max_tokens,
        temperature=temperature, top_p=top_p,
        frequency_penalty=frequency_penalty, presence_penalty=presence_penalty)
    return r.choices[0].message.content.strip()

def call_with_retry(client, model, uri, prompt, system_prompt=None,
                    max_retries=5, retry_delay=10.0, request_delay=2.0,
                    max_consecutive_429=5, **kw):
    consecutive = 0
    for attempt in range(1, max_retries + 1):
        try:
            result = call_api(client, model, uri, prompt, system_prompt=system_prompt, **kw)
            if request_delay > 0: time.sleep(request_delay)
            return result
        except Exception as e:
            err = str(e)
            if "429" in err:
                consecutive += 1
                w = min(retry_delay * attempt, 30.0)
                logging.warning("429 rate-limit x%d. Wait %.1fs (%d/%d)", consecutive, w, attempt, max_retries)
                if consecutive >= max_consecutive_429:
                    logging.error("Too many 429s — pausing 120s")
                    time.sleep(120)
                    consecutive = 0
            else:
                w = retry_delay
                logging.warning("API fail (%d/%d): %s", attempt, max_retries, e)
            time.sleep(w)
    logging.error("API failed after %d retries — returning empty", max_retries)
    return ""

# ============================================================
# Annotation loading
# ============================================================
def load_annotations(ann_path, dataset):
    data = json.loads(Path(ann_path).read_text(encoding="utf-8"))
    if isinstance(data, dict): data = data.get("annotations", data.get("data", []))
    normalized = []
    for item in data:
        if dataset == "kvasir":
            n = {"image_id":item.get("image_id",""),"image_path":item.get("image_path",""),
                 "modality":item.get("modality","endoscopy"),"tasks":item.get("tasks",{})}
        elif dataset == "slake":
            if "question" in item:
                n = {"image_id":item.get("img_name",""),"image_path":item.get("img_name",""),
                     "modality":item.get("modality","X-ray/CT/MRI"),
                     "tasks":{"vqa":[{"question":item["question"],"answer":item["answer"]}]}}
            else:
                n = {"image_id":item.get("folder_name",""),"image_path":item.get("folder_name",""),
                     "modality":"X-ray/CT/MRI","tasks":{"grounding":item.get("grounded_caption","")}}
        elif dataset == "radvqa":
            n = {"image_id":item.get("image_name",""),"image_path":item.get("image_name",""),
                 "modality":"X-ray","tasks":{"vqa":[{"question":item["question"],"answer":item["answer"]}]}}
        elif dataset == "indiana":
            _tasks = item.get("tasks", {})
            n = {"image_id":item.get("image_path",""),"image_path":item.get("image_path",""),
                 "modality":"X-Ray",
                 "tasks":{"report": _tasks.get("report", item.get("caption",""))}}
        elif dataset == "group_breast_us":
            n = {"image_id":item.get("image_id",""),"image_path":item.get("image_path",""),
                 "modality":item.get("modality","ultrasound"),"tasks":item.get("tasks",{})}
        else:
            n = item
        normalized.append(n)

    # Merge flat VQA entries that share the same image_id into one with grouped vqa[]
    if dataset in ("slake", "radvqa"):
        merged = {}
        for item in normalized:
            iid = item["image_id"]
            if iid not in merged:
                merged[iid] = {k: v for k, v in item.items() if k in ("image_id", "image_path", "modality")}
                merged[iid]["tasks"] = {}
            # Accumulate VQA pairs
            m_tasks = merged[iid].setdefault("tasks", {})
            vqa_list = m_tasks.setdefault("vqa", [])
            vqa_list.extend(item.get("tasks", {}).get("vqa", []))
            # Merge other task fields like grounding if present
            for key in ("grounding",):
                if key in item.get("tasks", {}) and key not in m_tasks:
                    m_tasks[key] = item["tasks"][key]
        normalized = list(merged.values())

    return normalized

# ============================================================
# Checkpoint: resume + incremental save
# ============================================================
def _unflatten_nested_checkpoint(data: list, task: str) -> list:
    """Convert nested [{image_id, image_path, modality, report/vqa: {...}}] 
    back to flat [{image_id, task, question, gt_answer, pred_answer}]."""
    flat = []
    task_key = "vqa" if task == "vqa" else task
    for item in data:
        iid = item.get("image_id", "")
        v = item.get(task_key)
        if v is None: continue
        # VQA may have a list of QA pairs per image
        if isinstance(v, list):
            v = v[0] if v else {}
        if not isinstance(v, dict): continue
        flat.append({
            "image_id": iid,
            "task": task,
            "question": v.get("question", ""),
            "gt_answer": v.get("gt_answer", ""),
            "pred_answer": v.get("pred_answer", ""),
        })
    return flat

def load_checkpoint(out_path: str, task: str) -> tuple[set, list]:
    if not os.path.exists(out_path): return set(), []
    try:
        data = json.loads(Path(out_path).read_text(encoding="utf-8"))
        if not isinstance(data, list): return set(), []
        ids = {item["image_id"] for item in data if "image_id" in item}
        logging.info("Resume: %d images already done", len(ids))
        return ids, _unflatten_nested_checkpoint(data, task)
    except Exception:
        logging.warning("Cannot load %s, starting fresh", out_path)
        return set(), []

def save_checkpoint(results: list, out_path: str):
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    os.replace(tmp, out_path)

# ============================================================
# VQA generation
# ============================================================
def generate_vqa(ann_data, image_dir, client, model, args, cfg):
    results = []
    total = len(ann_data) if args.max_samples==0 else min(args.max_samples, len(ann_data))
    mt = args.max_tokens or cfg.get("max_tokens", 32)
    tmp = args.temperature if args.temperature is not None else cfg.get("temperature", 0.0)
    for idx, info in enumerate(ann_data[:total]):
        iid = info.get("image_id", str(idx))
        uri = encode_image_base64(resolve_image_path(info.get("image_path",""), image_dir))
        if uri is None: continue
        vqa_pairs = (info.get("tasks",{}) or {}).get("vqa") or []
        if not vqa_pairs: continue
        if args.batch_questions:
            results += _vqa_batch(uri, iid, vqa_pairs, idx, total, client, model, args, cfg, mt, tmp)
        else:
            results += _vqa_one_by_one(uri, iid, vqa_pairs, idx, total, client, model, args, cfg, mt, tmp, info)
        if args.request_delay > 0 and idx < total - 1:
            time.sleep(args.request_delay)
    return results

def _vqa_one_by_one(uri, iid, vqa_pairs, idx, total, client, model, args, cfg, mt, tmp, info):
    results, sp = [], cfg["system_prompt"]
    for qi, qa in enumerate(vqa_pairs):
        q, gt = qa.get("question",""), str(qa.get("answer",""))
        p = build_user_prompt(q, cfg, info)
        pred = call_with_retry(client, model, uri, p, system_prompt=sp,
            max_retries=args.max_retries, retry_delay=args.retry_delay, request_delay=args.request_delay,
            max_tokens=mt, temperature=tmp, top_p=args.top_p,
            frequency_penalty=args.frequency_penalty, presence_penalty=args.presence_penalty)
        results.append({"image_id":iid,"task":"vqa","question":p,"gt_answer":gt,"pred_answer":pred})
        logging.info("[%d/%d] %s Q%d: GT=%s | Pred=%s", idx+1,total,iid,qi+1,gt[:30],pred[:50])
    return results

def _vqa_batch(uri, iid, vqa_pairs, idx, total, client, model, args, cfg, mt, tmp):
    lines = ["[vqa] Answer with ONLY the answer. No explanation.\n"]
    for qi, qa in enumerate(vqa_pairs): lines.append(f"Q{qi+1}: {qa['question']}")
    bs = "You are a medical AI. Answer with ONLY the short answer. Output format: Q1: ans1\\nQ2: ans2\\n..."
    pred = call_with_retry(client, model, uri, "\n".join(lines), system_prompt=bs,
        max_retries=args.max_retries, retry_delay=args.retry_delay, request_delay=args.request_delay,
        max_tokens=max(len(vqa_pairs)*8, mt), temperature=tmp,
        top_p=args.top_p, frequency_penalty=args.frequency_penalty, presence_penalty=args.presence_penalty)
    parsed = {}
    for line in pred.splitlines():
        m = re.match(r'Q(\d+)[\s:：]+(.+)', line.strip())
        if m: parsed[int(m.group(1))] = m.group(2).strip()
    results = []
    for qi, qa in enumerate(vqa_pairs):
        a = parsed.get(qi+1, "") or (pred.strip()[:100] if len(vqa_pairs)==1 else "")
        results.append({"image_id":iid,"task":"vqa",
            "question":build_user_prompt(qa.get("question",""), cfg, {}),
            "gt_answer":str(qa.get("answer","")), "pred_answer":a})
    logging.info("[%d/%d] %s BATCH %d q: %d ok", idx+1,total,iid,len(vqa_pairs),
                 len([r for r in results if r["pred_answer"]]))
    return results

# ============================================================
# Report & Grounding generation
# ============================================================
def generate_report(ann_data, image_dir, client, model, args, cfg):
    results, total = [], len(ann_data) if args.max_samples==0 else min(args.max_samples, len(ann_data))
    sp, mt = cfg["system_prompt"], args.max_tokens or cfg.get("max_tokens", 256)
    tmp = args.temperature if args.temperature is not None else cfg.get("temperature", 0.1)
    for idx, info in enumerate(ann_data[:total]):
        iid = info.get("image_id", str(idx))
        uri = encode_image_base64(resolve_image_path(info.get("image_path",""), image_dir))
        if uri is None: continue
        gt = (info.get("tasks",{}) or {}).get("report")
        if not gt: continue
        pred = call_with_retry(client, model, uri, build_user_prompt(None, cfg, info), system_prompt=sp,
            max_retries=args.max_retries, retry_delay=args.retry_delay, request_delay=args.request_delay,
            max_tokens=mt, temperature=tmp, top_p=args.top_p,
            frequency_penalty=args.frequency_penalty, presence_penalty=args.presence_penalty)
        results.append({"image_id":iid,"task":"report","question":build_user_prompt(None, cfg, info),
                        "gt_answer":gt,"pred_answer":pred})
        logging.info("[%d/%d] %s report: GT=%s | Pred=%s", idx+1,total,iid,str(gt)[:40],pred[:50])
        if args.request_delay > 0 and idx < total - 1:
            time.sleep(args.request_delay)
    return results

def generate_grounding(ann_data, image_dir, client, model, args, cfg):
    results, total = [], len(ann_data) if args.max_samples==0 else min(args.max_samples, len(ann_data))
    sp, mt = cfg["system_prompt"], args.max_tokens or cfg.get("max_tokens", 256)
    tmp = args.temperature if args.temperature is not None else cfg.get("temperature", 0.0)
    for idx, info in enumerate(ann_data[:total]):
        iid = info.get("image_id", str(idx))
        uri = encode_image_base64(resolve_image_path(info.get("image_path",""), image_dir))
        if uri is None: continue
        gt = (info.get("tasks",{}) or {}).get("grounding","")
        if not gt: continue
        pred = call_with_retry(client, model, uri, build_user_prompt(None, cfg, info), system_prompt=sp,
            max_retries=args.max_retries, retry_delay=args.retry_delay, request_delay=args.request_delay,
            max_tokens=mt, temperature=tmp, top_p=args.top_p,
            frequency_penalty=args.frequency_penalty, presence_penalty=args.presence_penalty)
        results.append({"image_id":iid,"task":"grounding",
            "question":build_user_prompt(None, cfg, info),"gt_answer":gt,"pred_answer":pred})
        logging.info("[%d/%d] %s grounding: GT=%s | Pred=%s", idx+1,total,iid,str(gt)[:40],pred[:50])
        if args.request_delay > 0 and idx < total - 1:
            time.sleep(args.request_delay)
    return results

# ============================================================
# VQA nested reformatting
# ============================================================
def _reformat_to_nested(flat, ann_data, task):
    """Convert flat result rows to nested per-image format (like kvasir_test.json).
    VQA: [{"image_id":"a","vqa":[{"question","gt_answer","pred_answer"},...]}]
    Report: [{"image_id":"a","report":[{"gt_answer","pred_answer"},...]}]
    """
    img_info = {}
    for item in ann_data:
        img_info[item.get("image_id","")] = {
            "image_path": item.get("image_path",""),
            "modality": item.get("modality",""), "anatomy": item.get("anatomy",""),
        }
    grp, order = {}, []
    for r in flat:
        iid = r["image_id"]
        if iid not in grp: grp[iid] = []; order.append(iid)
        if task in ("vqa", "grounding"):
            grp[iid].append({"question": r.get("question",""), "gt_answer": r.get("gt_answer",""),
                             "pred_answer": r.get("pred_answer","")})
        else:  # report
            grp[iid].append({"gt_answer": r.get("gt_answer",""), "pred_answer": r.get("pred_answer","")})

    key = "vqa" if task == "vqa" else task
    return [{"image_id": iid, "image_path": img_info.get(iid,{}).get("image_path",""),
             "modality": img_info.get(iid,{}).get("modality",""),
             key: grp[iid][0] if len(grp[iid]) == 1 else grp[iid]}
            for iid in order]

# ============================================================
# Main
# ============================================================
def main():
    args = resolve_defaults(parse_args())
    cfg = get_prompt_config(args.dataset, args.task)
    ann = Path(args.ann_file)
    if not ann.exists(): sys.exit(f"Annotation not found: {ann}")
    ann_data_full = load_annotations(str(ann), args.dataset)
    logging.info("Dataset=%s | Task=%s | N=%d", args.dataset, args.task, len(ann_data_full))
    logging.info("Images: %s | Model: %s | Out: %s", args.image_dir, args.model_name, args.output)

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    done_ids, results = load_checkpoint(str(out), args.task)
    ann_data = [item for item in ann_data_full if item.get("image_id","") not in done_ids]
    if not ann_data:
        logging.info("All %d images already done.", len(done_ids))
        return
    logging.info("%d remaining (%d done)", len(ann_data), len(done_ids))

    from openai import OpenAI
    client = OpenAI(base_url=args.api_base_url, api_key=args.api_key)

    try:
        if args.task == "vqa":
            results += generate_vqa(ann_data, args.image_dir, client, args.model_name, args, cfg)
        elif args.task == "report":
            results += generate_report(ann_data, args.image_dir, client, args.model_name, args, cfg)
        elif args.task == "grounding":
            results += generate_grounding(ann_data, args.image_dir, client, args.model_name, args, cfg)
        else:
            sys.exit(f"Unknown task: {args.task}")
    except KeyboardInterrupt:
        logging.warning("Interrupted — %d entries", len(results))
    except Exception as e:
        logging.error("Fatal: %s — %d entries", e, len(results))
    finally:
        nested = _reformat_to_nested(results, ann_data_full, args.task)
        # Never overwrite existing data with an empty result list.
        if nested or not os.path.exists(str(out)):
            save_checkpoint(nested, str(out))
        else:
            logging.warning("No new entries; keeping existing %s", out)
        logging.info("Final: %d entries in %s", len(nested), out)

if __name__ == "__main__":
    main()
