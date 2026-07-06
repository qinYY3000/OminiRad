Python API and Evaluation Code for v2.0 and v1.0 releases of the VQA dataset.
===================
## VQA v2.0 release ##
This release consists of
- Real 
	- 82,783 MS COCO training images, 40,504 MS COCO validation images and 81,434 MS COCO testing images (images are obtained from [MS COCO website] (http://mscoco.org/dataset/#download))
	- 443,757 questions for training, 214,354 questions for validation and 447,793 questions for testing
	- 4,437,570 answers for training and 2,143,540 answers for validation (10 per question)

There is only one type of task
- Open-ended task

## VQA v1.0 release ##
This release consists of
- Real 
	- 82,783 MS COCO training images, 40,504 MS COCO validation images and 81,434 MS COCO testing images (images are obtained from [MS COCO website] (http://mscoco.org/dataset/#download))
	- 248,349 questions for training, 121,512 questions for validation and 244,302 questions for testing (3 per image)
	- 2,483,490 answers for training and 1,215,120 answers for validation (10 per question)
- Abstract
	- 20,000 training images, 10,000 validation images and 20,000 MS COCO testing images
	- 60,000 questions for training, 30,000 questions for validation and 60,000 questions for testing (3 per image)
	- 600,000 answers for training and 300,000 answers for validation (10 per question)

There are two types of tasks
- Open-ended task
- Multiple-choice task (18 choices per question)

## Requirements ##
- python 2.7
- scikit-image (visit [this page](http://scikit-image.org/docs/dev/install.html) for installation)
- matplotlib (visit [this page](http://matplotlib.org/users/installing.html) for installation)

## Files ##
./Questions
- For v2.0, download the question files from the [VQA download page](http://www.visualqa.org/download.html), extract them and place in this folder.
- For v1.0, both real and abstract, question files can be found on the [VQA v1 download page](http://www.visualqa.org/vqa_v1_download.html).
- Question files from Beta v0.9 release (123,287 MSCOCO train and val images, 369,861 questions, 3,698,610 answers) can be found below
	- [training question files](http://visualqa.org/data/mscoco/prev_rel/Beta_v0.9/Questions_Train_mscoco.zip)
	- [validation question files](http://visualqa.org/data/mscoco/prev_rel/Beta_v0.9/Questions_Val_mscoco.zip)
- Question files from Beta v0.1 release (10k MSCOCO images, 30k questions, 300k answers) can be found [here](http://visualqa.org/data/mscoco/prev_rel/Beta_v0.1/Questions_Train_mscoco.zip).

./Annotations
- For v2.0, download the annotations files from the [VQA download page](http://www.visualqa.org/download.html), extract them and place in this folder.
- For v1.0, for both real and abstract, annotation files can be found on the [VQA v1 download page](http://www.visualqa.org/vqa_v1_download.html).
- Annotation files from Beta v0.9 release (123,287 MSCOCO train and val images, 369,861 questions, 3,698,610 answers) can be found below
	- [training annotation files](http://visualqa.org/data/mscoco/prev_rel/Beta_v0.9/Annotations_Train_mscoco.zip)
	- [validation annotation files](http://visualqa.org/data/mscoco/prev_rel/Beta_v0.9/Annotations_Val_mscoco.zip)
- Annotation files from Beta v0.1 release (10k MSCOCO images, 30k questions, 300k answers) can be found [here](http://visualqa.org/data/mscoco/prev_rel/Beta_v0.1/Annotations_Train_mscoco.zip).

./Images
- For real, create a directory with name mscoco inside this directory. For each of train, val and test, create directories with names train2014, val2014 and test2015 respectively inside mscoco directory, download respective images from [MS COCO website](http://mscoco.org/dataset/#download) and place them in respective folders.
- For abstract, create a directory with name abstract_v002 inside this directory. For each of train, val and test, create directories with names train2015, val2015 and test2015 respectively inside abstract_v002 directory, download respective images from [VQA download page](http://www.visualqa.org/download.html) and place them in respective folders.

./PythonHelperTools
- This directory contains the Python API to read and visualize the VQA dataset
- vqaDemo.py (demo script)
- vqaTools (API to read and visualize data)

./PythonEvaluationTools
- This directory contains the Python evaluation code
- vqaEvalDemo.py (evaluation demo script)
- vqaEvaluation (evaluation code)

./Results
- OpenEnded_mscoco_train2014_fake_results.json (an example of a fake results file for v1.0 to run the demo)
- Visit [VQA evaluation page] (http://visualqa.org/evaluation) for more details.

./QuestionTypes
- This directory contains the following lists of question types for both real and abstract questions (question types are unchanged from v1.0 to v2.0). In a list, if there are question types of length n+k and length n with the same first n words, then the question type of length n does not include questions that belong to the question type of length n+k.
- mscoco_question_types.txt
- abstract_v002_question_types.txt

## References ##
- [VQA: Visual Question Answering](http://visualqa.org/)
- [Microsoft COCO](http://mscoco.org/)

## Developers ##
- Aishwarya Agrawal (Virginia Tech)
- Code for API is based on [MSCOCO API code](https://github.com/pdollar/coco).
- The format of the code for evaluation is based on [MSCOCO evaluation code](https://github.com/tylin/coco-caption).

---

## OmniRad Project Directory Structure

The VQA evaluation tools above are reused by OmniRad for medical VQA evaluation
(VQA-RAD and SLAKE VQA).  Below is the canonical on-disk layout for the entire
project.  All paths in YAML configs are **relative to repo root** and resolved
at runtime by `_resolve_path()`.

```
OminiRad/                              ← repo root (= /home/cwq/MedicalDP/OminiRad on server)
├── data/                               ← all datasets (images + annotations)
│   ├── mimic_cxr/
│   │   └── image/                      ← MIMIC-CXR JPGs
│   ├── radvqa/
│   │   └── imgs/                       ← VQA-RAD images
│   ├── slake/
│   │   └── imgs/                       ← SLAKE images (shared by grounding + VQA)
│   ├── nlst/
│   │   └── NLST_images/                ← NLST CT slices (PNG)
│   ├── rsna/
│   │   └── RSNA-bbox-1024/             ← RSNA chest X-rays
│   ├── group_breast/                   ← private breast US
│   │   ├── frames/<pid>/<study>/<frame>.png
│   │   ├── masks/<pid>/<study>/<frame>.png
│   │   └── reports/<pid>/<study>.txt
│   ├── group_thyroid/                  ← private thyroid US
│   │   ├── frames/...
│   │   ├── masks/...
│   │   └── reports/...
│   └── annotations/                    ← all JSON annotation files
│       ├── MIMIC_train.json            ← 171,085 records (image_id, caption)
│       ├── MIMIC_test.json             ← 43,454 records
│       ├── NLST_train.json             ← 7,625 records (key, bbox)
│       ├── NLST_test.json              ← 1,654 records
│       ├── RSNA_train.json             ← 974 records
│       ├── RSNA_test.json              ← 244 records
│       ├── vqa_train.json              ← 1,796 records (VQA-RAD: image_name, question, answer)
│       ├── vqa_test.json               ← 452 records
│       ├── VQA_train_SLAKE.json        ← 8,165 records (SLAKE VQA: img_name, question, answer)
│       ├── VQA_test_SLAKE.json         ← 2,094 records
│       ├── grounding_train_SLAKE.json  ← 463 records (folder_name, grounded_caption)
│       ├── grounding_test_SLAKE.json   ← 116 records
│       ├── group_breast_train.json     ← generated by tools/build_unified_dataset.py
│       ├── group_breast_val.json
│       ├── group_breast_test.json
│       ├── group_thyroid_train.json
│       ├── group_thyroid_val.json
│       └── group_thyroid_test.json
├── weights/                            ← model weights (not in git)
│   ├── llama-2-7b-chat-hf/             ← LLaMA-2-7B-Chat HuggingFace checkpoint
│   ├── minigpt_med_pretrained.pth      ← MiniGPT-Med released checkpoint
│   └── sam_vit_h_4b8939.pth            ← SAM ViT-H weights
├── experiments/                        ← training & evaluation outputs (auto-created)
├── minigpt4/                           ← source code
│   ├── configs/
│   │   ├── models/
│   │   │   ├── minigpt_v2.yaml
│   │   │   └── omnirad.yaml            ← OmniRad default model config
│   │   └── datasets/
│   │       ├── mimic_cxr/
│   │       ├── radvqa/
│   │       ├── nlst/
│   │       ├── rsna/
│   │       ├── grounding_SLAKE/
│   │       ├── slake_vqa/              ← NEW: SLAKE VQA config
│   │       │   └── slake_vqa.yaml
│   │       ├── group_breast_us/
│   │       ├── group_thyroid_us/
│   │       └── ...
│   ├── models/
│   │   └── omnirad.py                  ← OmniRad model
│   ├── datasets/
│   │   └── datasets/
│   │       ├── mimic_cxr_dataset.py
│   │       ├── radvqa_dataset.py       ← VQA-RAD (reads image_name)
│   │       ├── SLAKE_dataset.py        ← GroundingSLAKEDatase + SlakeVQADataset (reads img_name)
│   │       ├── nlst_dataset.py
│   │       ├── rsna_dataset.py
│   │       ├── unified_us_dataset.py   ← Group US with structured supervision
│   │       └── structured_fields.py    ← default fields for legacy datasets
│   └── common/vqa_tools/VQA/           ← VQA eval tools (this directory)
├── train_configs/
│   └── omnirad_finetune.yaml           ← OmniRad training config (8 datasets)
├── eval_configs/
│   ├── minigptv2_eval.yaml
│   └── minigptv2_benchmark_evaluation.yaml
├── eval_scripts/
│   ├── model_evaluation.py
│   └── metrics.py
├── tools/
│   ├── build_unified_dataset.py
│   ├── dataset_utils.py
│   └── converters/
├── train.py
└── README.md
```

### Dataset → Task → JSON mapping

| Dataset (builder name) | Task | JSON file | Key fields | Records (train) |
|------------------------|------|-----------|------------|----------------:|
| `mimic_cxr` | Report generation | `MIMIC_train.json` | `image_id, caption` | 171,085 |
| `radvqa` | VQA (VQA-RAD) | `vqa_train.json` | `image_name, question, answer` | 1,796 |
| `slake_vqa` | VQA (SLAKE) | `VQA_train_SLAKE.json` | `img_name, question, answer` | 8,165 |
| `grounding_SLAKE` | Grounded caption | `grounding_train_SLAKE.json` | `folder_name, grounded_caption` | 463 |
| `nlst` / `refer_nlst` / `identify_nlst` | Detection / Refer / Identify | `NLST_train.json` | `key, bbox` | 7,625 |
| `rsna` / `refer_rsna` / `identify_rsna` | Detection / Refer / Identify | `RSNA_train.json` | `key, bbox` | 974 |
| `group_breast_us` | Report + Seg + Det + VQA | `group_breast_train.json` | `image_path, tasks{boxes, masks, K}` | 4,875 |
| `group_thyroid_us` | Report + Seg + Det + VQA | `group_thyroid_train.json` | `image_path, tasks{boxes, masks, K}` | 8,465 |
