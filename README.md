# OmniRad: A Unified Vision-Language Model for Multi-task Radiology

<p align="center">
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0.0-ee4c2c.svg" alt="PyTorch"></a>
  <a href="https://huggingface.co/meta-llama/Llama-2-7b-chat-hf"><img src="https://img.shields.io/badge/LLM-LLaMA--2--7B--Chat-0078D4.svg" alt="LLaMA-2"></a>
</p>

OmniRad is a unified multimodal medical framework that extends LLM-based vision-language models into structured clinical reasoning and dense prediction tasks. Built on [MiniGPT-Med](https://github.com/Vision-CAIR/MiniGPT-Med) and inspired by [LISA](https://github.com/dvlab-research/LISA)'s embedding-as-mask paradigm, OmniRad introduces:

- **AMU-Seg** — Adaptive Multi-query Segmentation with Uncertainty estimation
- **SAR-Loc** — Scale-Adaptive, Anatomy-Referenced Localization
- **Cross-task Consistency** — Geometric and semantic regularization across tasks

OmniRad supports **7 medical tasks** in a single model: report generation, VQA, detection, grounding, referring, identification, and segmentation — across **4 modalities** (X-ray, MRI/CT, ultrasound, endoscopy).

> **Datasets used in this project (6):**
> Indiana CXR (Open-i), VQA-RAD, SLAKE VQA, SLAKE Grounding, Group-Breast US, Kvasir (colonoscopy polyps).
> MIMIC-CXR and NLST were removed (replaced by Indiana CXR + Group-Breast US + Kvasir).

---

## Table of Contents

- [Installation](#installation)
- [Pretrained Weights](#pretrained-weights)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Demo](#demo)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Installation

```bash
git clone https://github.com/your-org/OmniRad.git
cd OmniRad
conda env create -f environment.yml
conda activate llama
```

---

## Pretrained Weights

Download the following weights and place them under `weights/`:

| Weight | Role | Source | Destination |
|--------|------|--------|-------------|
| LLaMA-2-7B-Chat | LLM backbone | [HuggingFace](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) | `weights/llama-2-7b-chat-hf/` |
| MiniGPT-Med checkpoint | language-vision initialization | [Google Drive](https://drive.google.com/file/d/1kjGLk6s9LsBmXfLWQFCdlwF3aul08Cl8/view) | `weights/minigpt_med_pretrained.pth` |
| **MedSAM ViT-B (default)** | medical segmentation backbone | [Google Dirve](https://drive.google.com/file/d/1hu0cpKT96G9apYbTb85TREewpoDv-4Hb/view?usp=drive_link) | `weights/medsam_vit_b.pth` |
| SAM ViT-B | raw SAM baseline (same backbone family as MedSAM) | [Meta](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) | `weights/sam_vit_b_01ec64.pth` |

```bash
mkdir -p weights
git clone https://huggingface.co/meta-llama/Llama-2-7b-chat-hf weights/llama-2-7b-chat-hf
wget -O weights/minigpt_med_pretrained.pth "https://drive.google.com/uc?export=download&id=1kjGLk6s9LsBmXfLWQFCdlwF3aul08Cl8"
# Default dense segmentation backbone
# Download the MedSAM checkpoint to weights/medsam_vit_b.pth
# Optional raw-SAM baselines
wget -O weights/sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

The default OmniRad configuration already points to **MedSAM**:

```yaml
dense_encoder:
  name: "medsam_vit_b"
  weights: "weights/medsam_vit_b.pth"
```

Notes:
- `sam_vit_b_01ec64.pth` is the official **SAM ViT-B** checkpoint, but it is **not** the strongest overall SAM checkpoint; within vanilla SAM, larger backbones usually perform better (`vit_h` > `vit_l` > `vit_b`) at higher memory cost.
- `MedSAM` is a **medical-domain fine-tuned checkpoint built on top of SAM ViT-B**. If your goal is medical segmentation, you should load `weights/medsam_vit_b.pth` directly rather than loading vanilla `sam_vit_b_01ec64.pth` again on top of it.
- `sam_vit_b_01ec64.pth` is still useful as an ablation / baseline when you want to compare `raw SAM ViT-B` vs `MedSAM ViT-B` under the same OmniRad code path.

---

## Dataset Preparation

### Directory Structure

All paths in YAML configs are **relative to the repository root** and resolved automatically at runtime.

Two kinds of data live under `data/`:

1. **Raw source data** (downloaded as-is; the model never reads these directly)
   - `data/Chest X/` — Open-i CSV files + `images/`
   - `data/radvqa/`, `data/slake/`, `data/rsna/`, `data/group_breast/`, `data/kvasir/` — image folders
2. **Training/eval annotations** (the only files the model loads)
   - `data/annotations/*.json` — produced by running the converter on the raw source

The converter (`tools/build_unified_dataset.py`) reads (1) and writes (2);
training and evaluation only touch (2) plus the image folders under (1).

```
OmniRad/
├── data/
│   ├── Chest X/                      # Indiana University Open-i — raw source
│   │   ├── images/                   #   *.dcm.png frames (download separately)
│   │   ├── indiana_projections.csv   #   uid → filename mapping      (raw)
│   │   └── indiana_reports.csv       #   uid → report (findings + impression) (raw)
│   ├── radvqa/imgs/                  # VQA-RAD images
│   ├── slake/imgs/                   # SLAKE images (shared by grounding + VQA)
│   ├── rsna/RSNA-bbox-1024/          # RSNA chest X-rays (zero-shot eval)
│   ├── group_breast/                 # Private breast US — raw source
│   │   ├── frames/<pid>/<study>/<frame>.png
│   │   ├── masks/<pid>/<study>/<frame>.png
│   │   └── reports/<pid>/<study>.txt
│   ├── kvasir/                       # Kvasir-SEG colonoscopy polyps
│   │   ├── imgs/                     #   original frames
│   │   ├── masks/                    #   instance masks (1-to-1 with images)
│   │   └── kavsir_bboxes.json        #   raw bbox annotations
│   └── annotations/                  # ★ The only JSON files loaded at train/eval time
├── weights/                          # Model weights (not tracked by git)
├── experiments/                      # Training outputs (auto-created)
└── ...
```

> The CSV files under `data/Chest X/` are **source data for the converter only** —
> they are consumed once by `tools/build_unified_dataset.py --datasets indiana`
> to produce `data/annotations/indiana_{train,test}.json`, and are never opened
> again during training or evaluation.  The same applies to `reports/<study>.txt`
> under the ultrasound dataset folders.

### Datasets in use

The training mixture and evaluation pipeline cover **6 datasets** spanning
**5 modalities** (chest X-ray, mixed X-ray/CT/MRI, ultrasound, endoscopy) and **7
task types**. Each dataset's exact task coverage is listed below:
[json path](https://drive.google.com/drive/folders/1-C_EjcF5bm9JHFbb704zKbt_o52Vl-WN?usp=drive_link)

| Dataset | Modality | Tasks (task tag) | Train / Val / Test | Source |
|---------|----------|------------------|-------------------|--------|
| **Indiana CXR (Open-i)** | Chest X-ray | Report generation `[report]` | 5,635 / 293 / 1,498 | [IU-Rays](https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university?resource=download-directory) |
| **VQA-RAD**              | X-ray / CT / MRI | Visual QA `[vqa]` | 1,706 / 90 / 452 | [VQA-RAD](https://www.modelscope.cn/datasets/ZEROCX/VQA-RAD) |
| **SLAKE VQA**            | X-ray / CT / MRI | Visual QA `[vqa]` | 7,757 / 408 / 2,094 | [med-vqa.com](https://www.med-vqa.com/slake/) |
| **SLAKE Grounding**      | X-ray / CT / MRI | Grounded caption `[grounding]` | 440 / 23 / 116 | same as above |
| **RSNA**                 | Chest X-ray | Pneumonia detection `[detection]` | 9,077 / 478 / 244 | [RSNA 2018](https://www.kaggle.com/datasets/sovitrath/rsna-pneumonia-detection-2018) |
| **Group-Breast US**      | Ultrasound | Segmentation • Detection • Report • Refer • Identify | ~4,875 / ~243 / ~488 | Internal |
| **Kvasir**              | Endoscopy (colonoscopy) | Segmentation • Detection • Refer • Identify • VQA | ~760 / ~40 / ~200 | [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/) |

> Every dataset ships `train/val/test` three-way splits.  The **val** split feeds best-checkpoint selection during training (`valid_splits: ["val"]`).  Public datasets use a random 95/5 split from their original train set; self-owned datasets use a patient-level 76/4/20 split.

> **Task tags** are the literal prefix prepended to the instruction string at
> training/inference time (e.g. `[INST] <Img>...</Img> [report] Describe this
> image in detail [/INST]`).  They drive token routing inside OmniRad and let
> the model know which task head should be activated.

> **Removed datasets:** MIMIC-CXR (replaced by Indiana CXR) and NLST (raw
> images unavailable). Their code, JSON files and configs have been deleted.

#### Per-task summary

| Task tag | Datasets contributing supervision |
|----------|-----------------------------------|
| `[report]` (report generation) | Indiana CXR, Group-Breast US |
| `[vqa]` (visual question answering) | VQA-RAD, SLAKE VQA, Kvasir |
| `[grounding]` (grounded caption with bbox) | SLAKE Grounding |
| `[detection]` (bbox prediction, `<BOX_S/M/L>`) | RSNA, Group-Breast US, Kvasir |
| `[refer]` (NL query → location, `<LOC>`) | Group-Breast US, Kvasir |
| `[identify]` (location → object label) | Group-Breast US, Kvasir |
| `[segmentation]` (pixel mask, `<SEG>`) | Group-Breast US, Kvasir |

### Task Prompt Templates

All current tasks are wrapped as:

```text
[INST] <Img><ImageHere></Img> {task_prompt} [/INST]
```

The active task prompts in this repository are:

| Task | Current prompt template(s) | Main dataset(s) | Expected output style |
|------|----------------------------|-----------------|-----------------------|
| Report | `[report] Describe this ultrasound image in detail.` | Group-Breast US | free-text report |
| VQA | `[vqa] {question}` | VQA-RAD, SLAKE VQA, Kvasir | short free-text answer |
| Grounding | `[grounding] please describe this image in details` / `[grounding] describe this image as detailed as possible` / `[grounding] summarize this image in details` / `[grounding] give a thorough description of what you see in this image` | SLAKE Grounding | grounded caption |
| Detection | `[detection] pneumonia` (RSNA) / `[detection] Locate all visible lesions or nodules.` (US) / `[detection] Locate all visible polyps.` (Kvasir) | RSNA, Group-Breast US, Kvasir | `<BOX_S>`, `<BOX_M>`, `<BOX_L>` tokens or bbox text |
| Refer | `[refer] Where is the lesion or nodule located?` / `[refer] Where is the polyp located?` | Group-Breast US, Kvasir | location token / bbox answer |
| Identify | `[identify] what is at location {bbox}` / related paraphrases | Group-Breast US, Kvasir | class label plus optional bbox / region |
| Segmentation | `[segmentation] Segment all visible lesions or nodules.` / `[segmentation] Segment all visible polyps.` | Group-Breast US, Kvasir | `<SEG>` token sequence |

Design rule:
- **Task identity comes from the task tag**, not from the lesion word itself.
- So `[detection] tumor` and `[segmentation] tumor` are **not conflicting by default** as long as the task tag is preserved.
- If you later experiment with text-prompted segmentation (for example, MedSAM3-style concept prompts), keep the explicit task tag and avoid collapsing everything into a bare concept prompt like `tumor`.

### Indiana University Chest X-Ray (Open-i)

This dataset ships as two CSV files plus an `images/` folder (download
separately from NIH Open-i — `version_2`, ~14 GB):

```
data/Chest X/
├── images/                       # *.dcm.png frames (one per image_id)
├── indiana_projections.csv       # uid, filename, projection
└── indiana_reports.csv           # uid, MeSH, Problems, image,
                                  # indication, comparison, findings, impression
```

Run the converter once to materialize the unified-schema JSON splits
(`indiana_train.json` / `indiana_test.json`):

```bash
python tools/build_unified_dataset.py --datasets indiana
```

The converter:
1. Cleans Open-i's `XXXX` PHI placeholders (collapsed to `[REDACTED]`).
2. Concatenates `findings + impression` into a free-form `caption`.
3. Splits by `uid` (patient-level) **80 / 20** train / test, keeping both
   views (Frontal + Lateral) of one patient in the same split.
4. Drops records whose report is essentially empty.

### Private Ultrasound Data (Group Internal)

Place private US data under `data/group_breast/` and Kvasir images under `data/kvasir/imgs/`,
then run the conversion script:

```bash
python tools/build_unified_dataset.py --datasets group_breast kvasir
```

This generates:
- `data/annotations/group_breast_{train,test}.json`
- `data/annotations/kvasir_{train,val,test}.json`

Each ultrasound frame is supervised on **5 tasks**: segmentation, detection,
report generation, referring, identification.  Kvasir frames are supervised on
**5 tasks**: segmentation, detection, referring, identification, VQA.  VQA pairs
follow the question style of the [Kvasir metadata.csv](https://datasets.simula.no/kvasir-seg/)
and are derived from bbox count, scale and position (13 QAs per frame).
Patient-level **76 / 4 / 20** split is used (same convention as Indiana CXR).

> Private Group-Breast US dataset is for internal use only and is not released with the paper.
> Kvasir-SEG is a public dataset — images must be downloaded separately from [Simula](https://datasets.simula.no/kvasir-seg/).

---

## Training

### Quick Start

```bash
# 1. Activate environment
conda activate miniGPT-Med

# 2. (Optional) Login to Weights & Biases
wandb login

# 3. Launch distributed training (3 × RTX 3090)
torchrun --nproc-per-node 3 --master-port 8889 \
    train.py --cfg-path train_configs/omnirad_finetune.yaml
```

### Training Configuration

The default configuration (`train_configs/omnirad_finetune.yaml`) includes:

| Component | Value |
|-----------|-------|
| Model architecture | `omnirad` (LLaMA-2-7B + EVA-ViT-G + SAM ViT-H + LoRA) |
| Special tokens | `<SEG>`, `<BOX_S>`, `<BOX_M>`, `<BOX_L>`, `<LOC>` |
| Training datasets | 7 datasets (Indiana, VQA-RAD, SLAKE VQA, SLAKE Grounding, RSNA, Group-Breast US, Kvasir) |
| Batch size | 2 per GPU × 3 GPUs = 6 |
| Max epochs | 100 |
| Learning rate | 1e-5 (cosine with warmup) |
| Mixed precision | AMP (fp16) |
| Gradient clipping | 1.0 |
| Checkpoint keeping | Latest 3 + best |

#### Sample-ratio mix (temperature-2 task-balanced sampling, total ≈ 93)

| Dataset | sample_ratio | Notes |
|---------|--------------|-------|
| `indiana_cxr` | 25 | Primary report-generation supervisor |
| `radvqa` | 8 | Medical VQA |
| `slake_vqa` | 10 | Larger English VQA corpus |
| `grounding_SLAKE` | 10 | Grounded caption |
| `group_breast_us` | 17 | Covers 5 tasks |
| `kvasir` | 23 | Covers 5 tasks; multi-polyp K≥2 emphasis |

### WandB Logging

Training logs the following metrics to WandB:

| Metric | Description |
|--------|-------------|
| `loss` | Total loss (text + det + loc + seg + cons) |
| `loss_text` | Language modeling loss |
| `loss_det` | Detection loss (scale-aware L1) |
| `loss_loc` | Localization loss (bbox + anatomy) |
| `loss_seg` | Segmentation loss (BCE + Dice + uncertainty + cardinality) |
| `loss_cons` | Consistency loss (mask-box alignment) |
| `lr` | Learning rate |

### Training Stages

OmniRad follows a four-stage progressive training strategy:

| Stage | Trainable | Data | Purpose |
|-------|-----------|------|---------|
| 1. Vision-Language Alignment | Projection layer | Indiana CXR | Align visual features with LLM |
| 2. Instruction Tuning | Projection + LoRA | Indiana + RadVQA + SLAKE | Multi-task text training |
| 3. Dense Task Adaptation | + Token heads + Mask decoder | + Group US (5 tasks each) | Unlock `<SEG>` / `<BOX>` / `<LOC>` |
| 4. Joint Fine-tuning | All above | Full mixture + consistency | Cross-task agreement |

---

## Evaluation

### Benchmark Evaluation

#### OmniRad (our model)

```bash
# Evaluate on all public + private datasets
python eval_scripts/model_evaluation.py \
    --cfg-path eval_configs/omnirad_evaluation.yaml \
    --dataset indiana_cxr,radvqa,slake_vqa,rsna,SLAKE,group_breast_us,kvasir

# Evaluate a single dataset
python eval_scripts/model_evaluation.py \
    --cfg-path eval_configs/omnirad_evaluation.yaml \
    --dataset indiana_cxr

# Evaluate the private ultrasound dataset — runs all 5 tasks
python eval_scripts/model_evaluation.py \
    --cfg-path eval_configs/omnirad_evaluation.yaml \
    --dataset group_breast_us

# Evaluate the Kvasir polyp dataset — runs 5 tasks (seg/det/refer/identify/vqa)
python eval_scripts/model_evaluation.py \
    --cfg-path eval_configs/omnirad_evaluation.yaml \
    --dataset kvasir
```

#### Baseline models (comparison)

```bash
# MiniGPT-Med (primary baseline — uses existing minigpt_v2 architecture)
python eval_scripts/baseline_evaluation.py \
    --model minigpt_med \
    --cfg-path eval_configs/minigptv2_benchmark_evaluation.yaml \
    --dataset indiana_cxr,radvqa,slake_vqa,rsna,SLAKE \
    --output-dir eval_results/minigpt_med

# MiniGPT-v2 (pre-medical-finetuning baseline)
python eval_scripts/baseline_evaluation.py \
    --model minigpt_v2 \
    --cfg-path eval_configs/minigptv2_eval.yaml \
    --dataset indiana_cxr,radvqa,slake_vqa,rsna,SLAKE \
    --output-dir eval_results/minigpt_v2
```

#### Generating comparison tables

After running evaluations for all models, generate comparison CSV tables:

```bash
python eval_scripts/compare_results.py
```

This scans `eval_results/*/*/summary.json` and produces:
- `table1_report_generation.csv` — BERT-Sim / BLEU-4 / ROUGE-L / CheXbert-F1
- `table2_vqa.csv` — VQA BERT-Sim
- `table3_detection_grounding.csv` — IoU
- `table4_ultrasound_multitask.csv` — Dice / IoU / BERT-Sim / Acc (ultrasound + endoscopy)
- `table5_ablation.csv` — Ablation study (OmniRad variants)
- `all_results_flat.csv` — All results in flat format

See `eval_results/README.md` for the full directory structure.

### Metrics

| Task | Metric |
|------|--------|
| Report generation | BERT-Similarity, BLEU-4, ROUGE-L, CheXbert-F1 (14-class clinical) |
| VQA | BERT-Similarity |
| Detection / Grounding | IoU, GIoU |
| Referring | IoU |
| Identification | Accuracy (label exact-match) |
| Segmentation | Dice, Mask IoU |
| Reasoning Segmentation | gIoU, cIoU |
| K=0 (healthy) | False-positive `<SEG>` emission rate |

> **Report generation metrics** are computed via `evaluate_report_generation()`
> which runs all four metrics in one pass: BERT-Sim (semantic similarity),
> BLEU-4 (n-gram precision), ROUGE-L (LCS recall), and CheXbert-F1 (14-class
> clinical disease label agreement using a regex-based labeler with negation
> detection).  CheXbert-F1 is most meaningful for chest X-ray reports
> (Indiana CXR); for ultrasound reports it may be less informative but is
> reported for consistency.

### Evaluation Datasets

| Dataset | Task(s) | Train | Val | Test |
|---------|---------|-------|-----|------|
| Indiana CXR (Open-i) | Report generation `[report]` | 5,635 | 293 | 1,498 |
| VQA-RAD | VQA `[vqa]` | 1,706 | 90 | 452 |
| SLAKE VQA | VQA `[vqa]` | 7,757 | 408 | 2,094 |
| RSNA | Detection `[detection]` (pneumonia) | 9,077 | 478 | 244 |
| SLAKE Grounding | Grounded caption `[grounding]` | 440 | 23 | 116 |
| Group-Breast US | Seg / Det / Report / Refer / Identify | ~4,875 | ~243 | ~488 |
| Kvasir | Seg / Det / Refer / Identify / VQA | ~760 | ~40 | ~200 |

For ultrasound, the segmentation task triggers OmniRad's mask-decoder hook (`generate(..., return_masks=True)`); predicted masks are written to `experiments/eval/<dataset>_segmentation_pred_masks/<image_id>_seg<k>.png` and Dice is computed against the ground-truth instance masks listed in the unified-schema JSON.

---

## Demo

```bash
# MiniGPT-Med (original) demo
python demo_v2.py --cfg-path eval_configs/minigptv2_eval.yaml --gpu-id 0

# OmniRad demo (requires trained checkpoint)
python demo_v2.py --cfg-path eval_configs/omnirad_evaluation.yaml --gpu-id 0
```

---

## Model Architecture

```
                    ┌──────────────────────────────────────────┐
                    │              Input Image                  │
                    └──────────┬──────────────┬─────────────────┘
                               │              │
                    ┌──────────▼──────┐ ┌─────▼──────────────┐
                    │  EVA-ViT-G      │ │  SAM ViT-H          │
                    │  (frozen, 448)  │ │  (frozen, 1024)     │
                    └──────────┬──────┘ └─────┬──────────────┘
                               │              │
                    ┌──────────▼──────┐       │
                    │  Projection     │       │
                    │  (4-patch merge)│       │
                    └──────────┬──────┘       │
                               │              │
                    ┌──────────▼──────────────▼───────────────┐
                    │         LLaMA-2-7B + LoRA                 │
                    │  (task-aware token routing)              │
                    │                                          │
                    │  ① Text tokens: report / VQA / etc.    ──┼──→ "Heart size is normal..."
                    │  ② Special tokens: <SEG> <BOX_S/M/L>    │
                    │                         <LOC>            │
                    └────────────────┬─────────────────────────┘
                                     │
                    ┌────────────────▼─────────────────────────┐
                    │     Task-Specific Heads                    │
                    │                                           │
                    │  Text (inline)  │  (<SEG>)  │ (<BOX>) │  │
                    │                 │  Mask Dec │ BoxHead  │  │
                    │  Generated in   │  (SAM)    │ (S/M/L)  │  │
                    │  autoregressive │  +Uncert  │ +Anatomy │  │
                    │  decode step    │           │          │  │
                    └─────────────────────────────────────────┘
```

### Key Innovations

1. **AMU-Seg** (Adaptive Multi-query + Uncertainty)
   - Variable-length `<SEG>` emission (K ∈ {0, 1, 2, ...})
   - Heteroscedastic uncertainty estimation per pixel
   - Self-rejection on ambiguous cases

2. **SAR-Loc** (Scale-Adaptive, Anatomy-Referenced)
   - Scale-stratified `<BOX_S/M/L>` tokens with separate regression heads
   - Dual-coordinate `<LOC>` output (pixel bbox + anatomy region)
   - Bidirectional report ⇌ region linking

3. **Cross-task Consistency**
   - Mask–Box geometric consistency
   - Report–Detection semantic consistency
   - Grounding–Segmentation spatial consistency

---

## Results

> Results will be updated upon completion of experiments.

### Public Datasets

| Task | Dataset | MiniGPT-Med | OmniRad |
|------|---------|-------------|---------|
| Report (BERT-Sim) | Indiana CXR | TBD | TBD |
| Report (BLEU-4) | Indiana CXR | TBD | TBD |
| Report (ROUGE-L) | Indiana CXR | TBD | TBD |
| Report (CheXbert-F1) | Indiana CXR | TBD | TBD |
| VQA (BERT-Sim) | VQA-RAD | TBD | TBD |
| VQA (BERT-Sim) | SLAKE VQA | TBD | TBD |
| Detection (IoU) | RSNA (zero-shot) | TBD | TBD |
| Grounding (IoU) | SLAKE | TBD | TBD |

### Private Ultrasound & Endoscopy Datasets

Group-Breast US is supervised on **5 tasks**: Segmentation / Detection / Report Generation / Referring / Identification.
Kvasir is supervised on **5 tasks**: Segmentation / Detection / Referring / Identification / VQA.

| Task | Dataset | MiniGPT-Med | OmniRad |
|------|---------|-------------|---------|
| Segmentation (Dice) | Group-Breast US | N/A | TBD |
| Segmentation (Dice) | Kvasir | N/A | TBD |
| Detection (IoU) | Group-Breast US | N/A | TBD |
| Detection (IoU) | Kvasir | N/A | TBD |
| Report (BERT-Sim) | Group-Breast US | N/A | TBD |
| VQA (BERT-Sim) | Kvasir | N/A | TBD |
| Refer (IoU) | Group-Breast US | N/A | TBD |
| Refer (IoU) | Kvasir | N/A | TBD |
| Identify (Acc) | Group-Breast US | N/A | TBD |
| Identify (Acc) | Kvasir | N/A | TBD |

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{omnirad2026,
  title={OmniRad: A Unified Vision-Language Model for Multi-task Radiology},
  author={Your Name and Collaborators},
  journal={Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```

Please also cite the foundational work:

```bibtex
@article{alkhaldi2026minigptmed,
  title={MiniGPT-Med: Large Language Model as a General Interface for Radiology Diagnosis},
  author={Alkhaldi, Asma and Alnajim, Raneem and Alabdullatef, Layan and Alyahya, Rawan and Chen, Jun and Zhu, Deyao and Alsinan, Ahmed and Elhoseiny, Mohamed},
  journal={Transactions on Machine Learning Research (TMLR)},
  year={2026}
}

@inproceedings{lai2024lisa,
  title={LISA: Reasoning Segmentation via Large Language Model},
  author={Lai, Xin and Tian, Zhuotao and Chen, Yukang and Li, Yuheng and Yuan, Yukang and Liu, Shu and Jia, Jiaya},
  booktitle={CVPR},
  year={2024}
}
```

---

## Acknowledgements

This project builds upon the following open-source works:

- [MiniGPT-Med](https://github.com/Vision-CAIR/MiniGPT-Med) — Base medical VLM
- [MiniGPT-v2](https://github.com/Vision-CAIR/MiniGPT-4) — Multi-task VLM framework
- [LISA](https://github.com/dvlab-research/LISA) — Embedding-as-mask paradigm
- [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything) — Dense visual encoder
- [LLaMA-2](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) — Language model backbone
- [PEFT](https://github.com/huggingface/peft) — Parameter-efficient fine-tuning

Data sources used in this project:
- [NIH Open-i / Indiana University Chest X-Ray](https://openi.nlm.nih.gov/faq)
- [VQA-RAD](https://osf.io/89kps/)
- [SLAKE](https://www.med-vqa.com/slake/)
- [RSNA Pneumonia Detection Challenge](https://www.rsna.org/rsnai/ai-image-challenge/rsna-pneumonia-detection-challenge-2018)
- [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/) — colonoscopy polyp dataset with bounding-box annotations

---

## License

This project is licensed under the MIT License. See [LICENSE.txt](LICENSE.txt) for details.

The LLaMA-2 model is licensed under its own [license](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/blob/main/LICENSE). SAM weights are licensed under the [Apache 2.0 License](https://github.com/facebookresearch/segment-anything/blob/main/LICENSE).
