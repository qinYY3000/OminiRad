"""
Kvasir-SEG → OmniRad 统一格式转换
===================================
将 Kvasir-SEG 息肉数据集转为 OmniRad 格式的 JSON，支持:
  - 检测 (bbox + scale)
  - 分割 (mask)
  - VQA (模板问答)

输出格式与 OmniRad 的 kvasir_train.json / kvasir_test.json 完全一致。

用法:
  python tools/convert_kvasir.py \
    --kvasir-root E:/data/kvasir-seg \
    --output-dir D:/AIPro/OminiRad/data/annotations \
    --image-root data/kvasir/imgs
"""

import argparse
import json
import sys
from pathlib import Path
from PIL import Image
import numpy as np

# 添加 OmniRad 项目路径
OMNIRAD_ROOT = Path(__file__).resolve().parents[2]
if str(OMNIRAD_ROOT) not in sys.path:
    sys.path.insert(0, str(OMNIRAD_ROOT))

from tools.dataset_utils import (
    mask_to_bbox, bbox_scale, dump_json, patient_level_split, derive_vqa,
)


# ============================================================
# VQA 模板 (与 OmniRad kvasir_test.json 格式一致)
# ============================================================

def generate_vqa(num_polyps: int, boxes: list, image_size: list) -> list:
    """生成 Kvasir-SEG 风格的 VQA 问答对。

    参照 OmniRad kvasir_test.json 中的 VQA 格式，生成以下类型的问题:
    - 息肉数量
    - 是否有异常
    - 息肉类型
    - 检查方式
    - 息肉位置
    - 息肉大小
    - 器械数量
    """
    qa = []

    # 1. 数量问题
    qa.append({
        "question": "How many polyps are in the image?",
        "answer": str(num_polyps)
    })

    # 2. 发现数量
    qa.append({
        "question": "How many findings are present?",
        "answer": str(num_polyps)
    })

    # 3. 异常检测
    if num_polyps > 0:
        qa.append({
            "question": "Are there any abnormalities in the image? Check all that are present.",
            "answer": "polyp"
        })
        qa.append({
            "question": "Does this image contain any finding?",
            "answer": "yes"
        })
        qa.append({
            "question": "Have all polyps been removed?",
            "answer": "no"
        })
    else:
        qa.append({
            "question": "Are there any abnormalities in the image? Check all that are present.",
            "answer": "none"
        })
        qa.append({
            "question": "Does this image contain any finding?",
            "answer": "no"
        })

    # 4. 息肉类型
    if num_polyps > 0:
        qa.append({
            "question": "What type of polyp is present?",
            "answer": "polyp"
        })

    # 5. 检查方式
    qa.append({
        "question": "What type of procedure is the image taken from?",
        "answer": "colonoscopy"
    })

    # 6. 检测难度
    if num_polyps > 0:
        qa.append({
            "question": "Is this finding easy to detect?",
            "answer": "yes" if num_polyps == 1 else "some are easy, some are difficult"
        })

    # 7. 息肉大小
    if num_polyps > 0:
        size_parts = []
        for i, box in enumerate(boxes):
            scale = box.get("scale", "M")
            scale_text = {"S": "small", "M": "medium", "L": "large"}.get(scale, "medium")
            size_parts.append(f"polyp {i+1}: {scale_text}")
        qa.append({
            "question": "What is the size of the polyp?",
            "answer": "; ".join(size_parts) if len(size_parts) > 1 else size_parts[0]
        })

    # 8. 位置 (基于 bbox 在图片中的位置)
    if num_polyps > 0:
        positions = []
        w, h = image_size
        for box in boxes:
            x1, y1, x2, y2 = box["bbox"]
            cx = (x1 + x2) / 2 / w if w > 0 else 0.5
            cy = (y1 + y2) / 2 / h if h > 0 else 0.5

            horiz = "left" if cx < 0.33 else ("right" if cx > 0.66 else "center")
            vert = "upper" if cy < 0.33 else ("lower" if cy > 0.66 else "middle")

            if horiz == "center":
                pos = f"{vert}-center"
            else:
                pos = f"{horiz}"
            positions.append(pos)
        qa.append({
            "question": "Where in the image is the abnormality?",
            "answer": "; ".join(positions)
        })

    # 9. 器械问题
    qa.append({
        "question": "How many instruments are in the image?",
        "answer": "0"
    })
    qa.append({
        "question": "Are there any instruments in the image? Check all that are present.",
        "answer": "none"
    })

    return qa


# ============================================================
# 主转换函数
# ============================================================

def convert_kvasir(kvasir_root: str, output_dir: str, image_root: str = "data/kvasir/imgs"):
    """转换 Kvasir-SEG 数据集为 OmniRad 格式。

    Args:
        kvasir_root: Kvasir-SEG 根目录 (含 images/, masks/, kavsir_bboxes.json)
        output_dir: 输出目录
        image_root: JSON 中 image_path 的前缀 (OmniRad 项目内的相对路径)
    """
    kvasir_root = Path(kvasir_root)
    images_dir = kvasir_root / "images"
    masks_dir = kvasir_root / "masks"
    bbox_json = kvasir_root / "kavsir_bboxes.json"

    # 加载 bbox 标注
    with open(bbox_json) as f:
        bbox_data = json.load(f)
    print(f"Loaded {len(bbox_data)} bbox entries from {bbox_json}")

    records = []
    for img_file in sorted(images_dir.glob("*.jpg")):
        sid = img_file.stem

        # 加载图片获取尺寸
        try:
            img = Image.open(img_file)
            w, h = img.size
        except Exception as e:
            print(f"  Skip {sid}: {e}")
            continue

        # 查找 mask
        mask_file = masks_dir / f"{sid}.jpg"
        mask_path_str = f"{image_root.replace('imgs', 'masks')}/{sid}.jpg" if mask_file.exists() else None

        # 从 kavsir_bboxes.json 获取 bbox
        bbox_info = bbox_data.get(sid, {})
        raw_bboxes = bbox_info.get("bbox", [])

        # 构建 boxes 列表
        boxes = []
        for bb in raw_bboxes:
            bbox = [int(bb["xmin"]), int(bb["ymin"]), int(bb["xmax"]), int(bb["ymax"])]
            boxes.append({
                "class": "polyp",
                "bbox": bbox,
                "scale": bbox_scale(bbox, w, h),
                "anatomy_region": "colon",
            })

        # 如果没有 bbox 标注但有 mask，从 mask 推导
        if not boxes and mask_file.exists():
            bbox = mask_to_bbox(str(mask_file))
            if bbox:
                boxes.append({
                    "class": "polyp",
                    "bbox": bbox,
                    "scale": bbox_scale(bbox, w, h),
                    "anatomy_region": "colon",
                })

        K = len(boxes)

        # 生成 VQA
        vqa_pairs = generate_vqa(K, boxes, [w, h])

        # 构建 record
        record = {
            "image_id": sid,
            "image_path": f"{image_root}/{sid}.jpg",
            "modality": "endoscopy",
            "anatomy": "colon",
            "image_size": [w, h],
            "tasks": {
                "report": None,
                "vqa": vqa_pairs,
                "boxes": boxes,
                "masks": [
                    {"class": "polyp", "mask_path": mask_path_str}
                ] if mask_path_str else [],
                "K": K,
            },
            "patient_id": sid,
        }
        records.append(record)

    print(f"Converted {len(records)} samples")

    # 按患者级别切分 (80% train, 10% val, 10% test)
    train, val, test = patient_level_split(records, ratios=(0.8, 0.1, 0.1))

    # 保存
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dump_json(train, output_dir / "kvasir_train.json")
    dump_json(val, output_dir / "kvasir_val.json")
    dump_json(test, output_dir / "kvasir_test.json")

    print(f"\n{'='*60}")
    print(f"Kvasir-SEG → OmniRad conversion complete")
    print(f"{'='*60}")
    print(f"  Train: {len(train)} samples")
    print(f"  Val:   {len(val)} samples")
    print(f"  Test:  {len(test)} samples")
    print(f"  Output: {output_dir}")

    # 统计
    total_polyps = sum(r["tasks"]["K"] for r in records)
    print(f"\n  Total polyps: {total_polyps}")
    print(f"  Avg polyps/image: {total_polyps/len(records):.2f}")

    # scale 分布
    scale_counts = {"S": 0, "M": 0, "L": 0}
    for r in records:
        for box in r["tasks"]["boxes"]:
            scale_counts[box["scale"]] = scale_counts.get(box["scale"], 0) + 1
    print(f"  Scale distribution: {scale_counts}")

    return train, val, test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Kvasir-SEG to OmniRad format")
    parser.add_argument("--kvasir-root", type=str, default="E:/data/kvasir-seg",
                        help="Kvasir-SEG root directory")
    parser.add_argument("--output-dir", type=str, default="D:/AIPro/OminiRad/data/annotations",
                        help="Output directory for JSON files")
    parser.add_argument("--image-root", type=str, default="data/kvasir/imgs",
                        help="Image path prefix in JSON (relative to OmniRad project)")
    args = parser.parse_args()

    convert_kvasir(args.kvasir_root, args.output_dir, args.image_root)
