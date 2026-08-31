# 对比基线模型权重下载

所有权重统一存放在 `weights/{模型名称}/` 目录下。

## 1. BiomedCLIP（生物医学对比学习 VLM，仅 VQA）

- **HuggingFace**: `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`
- **论文**: BiomedCLIP: a multimodal biomedical foundation model pretrained on fifteen million scientific image-text pairs (MICCAI 2023)
- **权重路径**: `weights/biomedclip/`（需包含 `open_clip_pytorch_model.bin` 和 `open_clip_config.json`）
- **依赖**: `pip install open_clip_torch`
- **下载命令**:
  ```bash
  huggingface-cli download microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 \
    open_clip_pytorch_model.bin open_clip_config.json \
    --local-dir weights/biomedclip
  ```
- **支持任务**: VQA（零样本答案选择，对比学习无生成能力）
- **VQA 方式**: 对每个问题按问题类型生成候选答案集，计算图文相似度选最佳答案
- **加载方式**: 使用 `open_clip` 库（非 `transformers`），按官方 local-file 模式加载

## 2. CheXagent（胸片专用，3B）

- **HuggingFace**: `StanfordAIMI/CheXagent-2-3b`
- **论文**: CheXagent-2: Towards Large Language Models that can Read Medical Reports (2024)
- **权重路径**: `weights/chexagent/`
- **下载命令**:
  ```bash
  huggingface-cli download StanfordAIMI/CheXagent-2-3b --local-dir weights/chexagent
  ```
- **支持任务**: 报告生成（仅胸部 X 光）

## 3. LLaVA-Med（微软医学 VLM）

- **官方仓库**: https://github.com/microsoft/LLaVA-Med
- **论文**: LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day (NeurIPS 2024)
- **权重路径**: `weights/llava_med/`
- **下载命令**:
  ```bash
  # 从微软官方仓库下载 llava_med_weights 目录
  # 参考官方 README 的 Download 部分获取最新 checkpoint 链接
  # 将整个目录内容放到 weights/llava_med/
  ```
- **支持任务**: 报告生成、VQA

## 4. Med-Flamingo（医学少样本 VLM）

- **HuggingFace**: `med-flamingo/med-flamingo`
- **官方仓库**: https://github.com/sarahESL/MedFlamingo
- **论文**: Med-Flamingo: A Multimodal Medical Few-shot Learner (ML4H 2023)
- **权重路径**: `weights/med_flamingo/`（含 `model.pt` 和 `llama-7b/` 子目录）
- **下载命令**:
  ```bash
  # 1. 下载 med-flamingo checkpoint
  huggingface-cli download med-flamingo/med-flamingo model.pt --local-dir weights/med_flamingo

  # 2. 下载 llama-7b 基础权重（Med-Flamingo 依赖）
  huggingface-cli download huggyllama/llama-7b --local-dir weights/med_flamingo/llama-7b
  ```
- **支持任务**: 报告生成、VQA

## 5. RadFM（放射学基础模型）

- **官方仓库**: https://github.com/BoyiD/RadFM
- **论文**: RadFM: Radiology Foundation Model (MICCAI 2023)
- **权重路径**: `weights/radfm/pytorch_model.bin`
- **下载命令**:
  ```bash
  # 从官方仓库下载 pytorch_model.bin
  # 参考 https://github.com/BoyiD/RadFM 的 README 获取权重链接
  # 放到 weights/radfm/pytorch_model.bin
  ```
- **支持任务**: 报告生成、VQA

## 6. XrayGPT（胸片报告生成）

- **官方仓库**: https://github.com/mbzuai-oryx/XrayGPT
- **论文**: XrayGPT: Pioneering LLMs for Radiology Summarization and Understanding (2023)
- **权重路径**: `weights/xraygpt/`（含 `Vicuna_Radiology_fp16/` 子目录和 `xraygpt_pretrained1.pth`）
- **下载命令**:
  ```bash
  # 从官方仓库下载以下两个文件：
  # 1. Vicuna_Radiology_fp16/   → 放到 weights/xraygpt/Vicuna_Radiology_fp16/
  # 2. xraygpt_pretrained1.pth  → 放到 weights/xraygpt/xraygpt_pretrained1.pth
  # 参考 https://github.com/mbzuai-oryx/XrayGPT 的 README 获取下载链接
  ```
- **支持任务**: 报告生成（仅胸部 X 光）

## 7. MiniGPT-Med（OmniRad 直接基线）

- **论文**: MiniGPT-Med: A Unified Vision-Language Model for Radiology Image Understanding (TMLR 02/2026)
- **权重路径**: `weights/minigpt_med_pretrained.pth`
- **下载命令**:
  ```bash
  # 从 https://github.com/Vision-CAIR/MiniGPT-Med 下载官方权重
  # 放到 weights/minigpt_med_pretrained.pth
  ```
- **支持任务**: 报告生成、VQA、检测、grounding、identify

## 目录结构总览

```
weights/
├── biomedclip/                   # BiomedCLIP (PubMedBERT + ViT)
├── chexagent/                    # CheXagent-2-3b
├── llava_med/                    # LLaVA-Med 权重
├── med_flamingo/
│   ├── model.pt                  # Med-Flamingo checkpoint
│   └── llama-7b/                 # LLaMA-7B 基础权重
├── radfm/
│   └── pytorch_model.bin         # RadFM 权重
├── xraygpt/
│   ├── Vicuna_Radiology_fp16/    # 放射学微调 Vicuna
│   └── xraygpt_pretrained1.pth   # XrayGPT checkpoint
├── minigpt_med_pretrained.pth    # MiniGPT-Med 权重
└── llama-2-7b-chat-hf/            # LLaMA-2-7B（OmniRad/MiniGPT-Med 共用）
```

## 运行评估

下载完权重后，运行评估：

```bash
# BiomedCLIP VQA（仅 VQA，无报告生成能力）
python eval_scripts/baseline_evaluation.py \
  --model biomedclip \
  --dataset radvqa,slake_vqa \
  --output-dir eval_results/biomedclip

# CheXagent 报告生成
python eval_scripts/baseline_evaluation.py \
  --model chexagent \
  --dataset indiana_cxr \
  --output-dir eval_results/chexagent

# LLaVA-Med 报告 + VQA
python eval_scripts/baseline_evaluation.py \
  --model llava_med \
  --dataset indiana_cxr,radvqa,slake_vqa \
  --output-dir eval_results/llava_med

# Med-Flamingo 报告 + VQA
python eval_scripts/baseline_evaluation.py \
  --model med_flamingo \
  --dataset indiana_cxr,radvqa,slake_vqa \
  --output-dir eval_results/med_flamingo

# RadFM 报告 + VQA
python eval_scripts/baseline_evaluation.py \
  --model radfm \
  --dataset indiana_cxr,radvqa,slake_vqa \
  --output-dir eval_results/radfm

# XrayGPT 报告生成
python eval_scripts/baseline_evaluation.py \
  --model xraygpt \
  --dataset indiana_cxr \
  --output-dir eval_results/xraygpt
```
