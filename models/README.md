# 对比基线模型

所有权重统一存放在 `weights/` 目录下，每个模型一个子目录。

---

## 总览

| 模型 | 架构 | 参数 | 报告 | VQA | 权重获取难度 |
|------|------|------|:--:|:--:|:----------:|
| [BiomedCLIP](#1-biomedclip) | PubMedBERT + ViT (对比学习) | ~150M | ❌ | ✅ | ⭐ 简单 |
| [CheXagent](#2-chexagent) | InternViT + Vicuna | 3B | ✅ | ❌ | ⭐ 简单 |
| [LLaVA-Med](#3-llava-med) | CLIP ViT + LLaMA-7B | 7B | ✅ | ✅ | ⭐ 简单 |
| [Med-Flamingo](#4-med-flamingo) | CLIP ViT + LLaMA-7B (cross-attn) | 9B | ✅ | ✅ | ⭐⭐ 中等 |
| [XrayGPT](#5-xraygpt) | EVA ViT-G + Q-Former + Vicuna-7B | 7B | ✅ | ❌ | ⭐⭐⭐ 较难 |

---

## 1. BiomedCLIP

- **来源**: HuggingFace [`microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)
- **论文**: [BiomedCLIP (MICCAI 2023)](https://arxiv.org/abs/2303.00915)
- **依赖**: `pip install open_clip_torch`
- **任务**: VQA（对比学习，无生成能力）

```bash
huggingface-cli download microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 \
  open_clip_pytorch_model.bin open_clip_config.json \
  --local-dir weights/biomedclip
```

目录结构：
```
weights/biomedclip/
├── open_clip_pytorch_model.bin    # ~600 MB
└── open_clip_config.json
```

---

## 2. CheXagent

- **来源**: HuggingFace [`StanfordAIMI/CheXagent-2-3b`](https://huggingface.co/StanfordAIMI/CheXagent-2-3b)
- **论文**: [CheXagent-2 (2024)](https://arxiv.org/abs/2401.12208)
- **依赖**: `trust_remote_code=True`，自动处理
- **任务**: 报告生成（仅胸部 X 光）

```bash
huggingface-cli download StanfordAIMI/CheXagent-2-3b --local-dir weights/chexagent
```

目录结构：
```
weights/chexagent/            # 完整 HF 模型目录 (~6 GB)
├── config.json
├── model.safetensors
├── tokenizer_config.json
└── ...
```

---

## 3. LLaVA-Med

- **来源**: HuggingFace [`microsoft/llava-med-v1.5-mistral-7b`](https://huggingface.co/microsoft/llava-med-v1.5-mistral-7b)
- **论文**: [LLaVA-Med (NeurIPS 2024)](https://arxiv.org/abs/2306.00890)
- **依赖**: 自动下载 `openai/clip-vit-large-patch14`（~1.7 GB）
- **任务**: 报告生成、VQA

```bash
huggingface-cli download microsoft/llava-med-v1.5-mistral-7b --local-dir weights/llava_med
```

> **注意**: 当前代码中的 `models/llavamed.py` 使用的是 v1.0 版本（LLaMA-7B backbone）。v1.5 改用 Mistral-7B，需要相应修改加载代码才能使用。v1.0 权重需从[官方仓库 v1.0.0 分支](https://github.com/microsoft/LLaVA-Med/tree/v1.0.0)获取。

目录结构：
```
weights/llava_med/             # 完整 HF 模型目录 (~14 GB)
├── config.json
├── model-*.safetensors
├── tokenizer*
└── ...
```

附加自动下载（运行时）：
```
~/.cache/huggingface/hub/
└── models--openai--clip-vit-large-patch14/   # ~1.7 GB
```

---

## 4. Med-Flamingo

- **来源**: HuggingFace [`med-flamingo/med-flamingo`](https://huggingface.co/med-flamingo/med-flamingo)
- **论文**: [Med-Flamingo (ML4H 2023)](https://arxiv.org/abs/2307.15189)
- **官方仓库**: [snap-stanford/med-flamingo](https://github.com/snap-stanford/med-flamingo)
- **依赖**: `open-flamingo` 自动下载 CLIP ViT-L-14
- **任务**: 报告生成、VQA

```bash
# 1. Med-Flamingo checkpoint
huggingface-cli download med-flamingo/med-flamingo model.pt --local-dir weights/med_flamingo

# 2. LLaMA-7B（Med-Flamingo 依赖）
huggingface-cli download huggyllama/llama-7b --local-dir weights/med_flamingo/llama-7b
```

目录结构：
```
weights/med_flamingo/
├── model.pt                     # ~18 GB（Med-Flamingo delta 权重）
└── llama-7b/                    # ~26 GB（LLaMA-7B 完整权重）
    ├── config.json
    ├── pytorch_model-*.bin
    ├── tokenizer*
    └── ...
```

附加自动下载（运行时）：
```
~/.cache/huggingface/hub/
└── models--openai--clip-vit-large-patch14/   # ~1.7 GB
```

---

## 5. XrayGPT

- **官方仓库**: [mbzuai-oryx/XrayGPT](https://github.com/mbzuai-oryx/XrayGPT)
- **论文**: [XrayGPT (BIONLP-ACL 2024)](https://arxiv.org/abs/2306.07971)

需要下载 **两个文件**：

### 5.1 Vicuna-7B 放射学微调权重

来自官方 SharePoint（需有权限访问）：

```
https://mbzuaiac-my.sharepoint.com/:u:/g/personal/omkar_thawakar_mbzuai_ac_ae/EWoMYn3x7sdEnM2CdJRwWZgBCkMpLM03bk4GR5W0b3KIQQ
```

下载后解压放到 `weights/xraygpt/Vicuna_Radiology_fp16/`。

### 5.2 XrayGPT 最终检查点

来自官方 Google Drive：

```
https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R/view
```

下载后放到 `weights/xraygpt/xraygpt_pretrained1.pth`。

### 5.3 自动下载依赖

运行时 `timm` 会从 Google Storage 自动下载，缓存到 `~/.cache/torch/hub/checkpoints/`：

| 文件 | 来源 | 大小 |
|------|------|------|
| `eva_vit_g.pth` | [Google Storage](https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth) | ~4 GB |
| `blip2_pretrained_flant5xxl.pth` | [Google Storage](https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth) | ~5 GB |

### 目录结构

```
weights/xraygpt/
├── Vicuna_Radiology_fp16/        # Vicuna-7B 放射学微调 (~13 GB)
│   ├── config.json
│   ├── pytorch_model-*.bin
│   └── tokenizer*
└── xraygpt_pretrained1.pth      # XrayGPT checkpoint (~3 GB)

# 运行时自动下载到：
~/.cache/torch/hub/checkpoints/
├── eva_vit_g.pth                      # ~4 GB
└── blip2_pretrained_flant5xxl.pth     # ~5 GB
```

> **离线服务器**：如果无法访问 Google Storage，提前下载上述两个文件到本地 `~/.cache/torch/hub/checkpoints/` 即可。

---

## 6. MiniGPT-Med（OmniRad 直接基线，共用 LLaMA-2）

- **官方仓库**: [Vision-CAIR/MiniGPT-Med](https://github.com/Vision-CAIR/MiniGPT-Med)
- **论文**: [MiniGPT-Med (TMLR 2026)](https://arxiv.org/abs/2311.04112)

```bash
# 从官方仓库下载权重
curl -L -o weights/minigpt_med_pretrained.pth <官方下载链接>

# LLaMA-2-7B（与 OmniRad 共用）
huggingface-cli download meta-llama/Llama-2-7b-chat-hf --local-dir weights/llama-2-7b-chat-hf
```

---

## 最终目录结构

```
weights/
├── biomedclip/
│   ├── open_clip_pytorch_model.bin          # ~600 MB
│   └── open_clip_config.json
├── chexagent/                                # ~6 GB (HF 目录)
├── llava_med/                                # ~14 GB (HF 目录)
├── med_flamingo/
│   ├── model.pt                              # ~18 GB
│   └── llama-7b/                             # ~26 GB (HF 目录)
├── xraygpt/
│   ├── Vicuna_Radiology_fp16/                # ~13 GB
│   └── xraygpt_pretrained1.pth              # ~3 GB
├── minigpt_med_pretrained.pth                # ~3 GB
└── llama-2-7b-chat-hf/                       # ~26 GB（OmniRad/MiniGPT-Med 共用）
```

**总磁盘需求（不含 LLaMA-2）: ~57 GB**  
**含 LLaMA-2: ~83 GB**  
**含 LLaVA-Med v1.5: ~97 GB**

---

## 运行评估

```bash
# BiomedCLIP — VQA
python eval_scripts/baseline_evaluation.py --model biomedclip --dataset radvqa,slake_vqa -o eval_results/biomedclip

# CheXagent — 报告生成
python eval_scripts/baseline_evaluation.py --model chexagent --dataset indiana_cxr -o eval_results/chexagent

# LLaVA-Med — 报告 + VQA
python eval_scripts/baseline_evaluation.py --model llava_med --dataset indiana_cxr,radvqa,slake_vqa -o eval_results/llava_med

# Med-Flamingo — 报告 + VQA
python eval_scripts/baseline_evaluation.py --model med_flamingo --dataset indiana_cxr,radvqa,slake_vqa -o eval_results/med_flamingo

# XrayGPT — 报告生成
python eval_scripts/baseline_evaluation.py --model xraygpt --dataset indiana_cxr -o eval_results/xraygpt
```
