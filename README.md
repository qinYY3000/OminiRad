# OminiRad
OminiRad: A Unified Vision-Language Model for Multi-task Radiology

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
| **MedSAM ViT-B (default)** | medical segmentation backbone | Official MedSAM pretrained release | `weights/medsam_vit_b.pth` |
| SAM ViT-B | raw SAM baseline (same backbone family as MedSAM) | [Meta](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) | `weights/sam_vit_b_01ec64.pth` |
| SAM ViT-H | larger raw SAM baseline | [Meta](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) | `weights/sam_vit_h_4b8939.pth` |

```bash
mkdir -p weights
git clone https://huggingface.co/meta-llama/Llama-2-7b-chat-hf weights/llama-2-7b-chat-hf
wget -O weights/minigpt_med_pretrained.pth "https://drive.google.com/uc?export=download&id=1kjGLk6s9LsBmXfLWQFCdlwF3aul08Cl8"
# Default dense segmentation backbone
# Download the MedSAM checkpoint to weights/medsam_vit_b.pth
# Optional raw-SAM baselines
wget -O weights/sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
wget -O weights/sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
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
| **Indiana CXR (Open-i)** | Chest X-ray | Report generation `[report]` | 5,635 / 293 / 1,498 | [NIH Open-i](https://openi.nlm.nih.gov/faq) |
| **VQA-RAD**              | X-ray / CT / MRI | Visual QA `[vqa]` | 1,706 / 90 / 452 | [OSF](https://osf.io/89kps/) |
| **SLAKE VQA**            | X-ray / CT / MRI | Visual QA `[vqa]` | 7,757 / 408 / 2,094 | [med-vqa.com](https://www.med-vqa.com/slake/) |
| **SLAKE Grounding**      | X-ray / CT / MRI | Grounded caption `[grounding]` | 440 / 23 / 116 | same as above |
| **RSNA**                 | Chest X-ray | Pneumonia detection `[detection]` | 9,077 / 478 / 244 | [RSNA 2018](https://www.rsna.org/rsnai/ai-image-challenge/rsna-pneumonia-detection-challenge-2018) |
| **Group-Breast US**      | Ultrasound | Segmentation • Detection • Report • Refer • Identify | ~4,875 / ~243 / ~488 | Internal |
| **Kvasir**              | Endoscopy (colonoscopy) | Segmentation • Detection • Refer • Identify • VQA | ~760 / ~40 / ~200 | [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/) |
