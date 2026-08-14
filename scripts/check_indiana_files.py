"""核对 Indiana JSON 文件引用的图片是否都在磁盘上存在。"""
import json
import os

IMG_DIR = "/path/to/E/datasets/Chest X"  # 改成实际路径

img_files = set(os.listdir(IMG_DIR))

for split in ["train", "val", "test"]:
    path = f"data/annotations/indiana_{split}.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    json_names = set(item["filename"] for item in data)
    missing = json_names - img_files
    extra = img_files - json_names

    print(f"\n=== indiana_{split}.json ===")
    print(f"  JSON 记录: {len(data)}")
    print(f"  磁盘图片:  {len(img_files)}")
    print(f"  匹配:      {len(json_names & img_files)}")
    print(f"  JSON有但磁盘缺: {len(missing)}")
    print(f"  磁盘有但JSON无: {len(extra)}")
    if missing:
        print(f"  缺失样例: {sorted(missing)[:5]}")