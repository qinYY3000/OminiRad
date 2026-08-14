import json
import os

img_dir = r'E:\datasets\ChestX-Ray\images_normalized'
img_files = set(os.listdir(img_dir))

for split in ['train', 'val', 'test']:
    path = f'data/annotations/indiana_{split}.json'
    with open(path, encoding='utf-8') as f:
        data = json.load(f)

    filtered = []
    for item in data:
        filename = item.get('filename', f'{item["image_id"]}.png')
        if filename in img_files:
            # Normalize backslashes to forward slashes
            item['image_path'] = item['image_path'].replace('\\', '/')
            filtered.append(item)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)

    print(f'indiana_{split}.json: {len(data)} -> {len(filtered)} records')