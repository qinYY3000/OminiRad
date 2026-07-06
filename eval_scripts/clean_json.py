import json
import re


def clean_report_json(messy_json, cleaned_output):
    """Normalize an inference dump for any report-generation dataset.

    Input shape (written by ``model_evaluation.process_*``)::

        { "<image_id>.<ext>": ["caption 1", "caption 2", ...], ... }

    Output shape (consumed by ``report_bert_sim``)::

        [{"image_id": "<image_id>", "caption": "concatenated"}, ...]
    """
    with open(messy_json, "r") as f:
        messy_data = json.load(f)

    clean_data = []
    for image_id, captions in messy_data.items():
        image_id_clean = image_id.split(".")[0]
        caption_clean = " ".join(captions)
        clean_data.append({
            "image_id": image_id_clean,
            "caption": caption_clean,
        })

    with open(cleaned_output, "w") as outfile:
        json.dump(clean_data, outfile, indent=2)


def clean_vqa_json(messy_json, cleaned_output):
    with open(messy_json, "r") as file:
        messy_json = json.load(file)

    organized_json = {}

    for key, values in messy_json.items():
        organized_json[key] = []
        for value in values:
            organized_json[key].append({
                "question": value["question"],
                "answer": value["answer"]
            })

    with open(cleaned_output, "w") as outfile:
        json.dump(organized_json, outfile, indent=4)



def clean_detection_json(messy_json, cleaned_output):

    with open(messy_json, "r") as input_file:
        input_json = json.load(input_file)

    organized_data = []

    for key, value in input_json.items():
        if value and isinstance(value, list) and len(value) > 0:
            caption = value[0]
            objects_match = caption.split("<p>")
            if len(objects_match) == 2:
                object_part = objects_match[1].split("</p>")[0].strip()
            else:
                object_part = ""
            
            bbox_match = re.findall(r'<(\d+)>', caption)
            
            if object_part and bbox_match and len(bbox_match) == 4:
                key_part = key.split(".png")[0]
                bbox_values = [float(val) for val in bbox_match]

                organized_item = {
                    "key": key_part,
                    "objects": [object_part],
                    "bbox": [bbox_values],
                }

                organized_data.append(organized_item)

    with open(cleaned_output, "w") as output_file:
        json.dump(organized_data, output_file, indent=4)