import json
import os
import random
from PIL import Image
from torch.utils.data import Dataset
from minigpt4.datasets.datasets.structured_fields import add_default_structured_fields

class GroundingSLAKEDatase(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path):

        self.vis_root = vis_root
        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self.instruction_pool = [
            '[grounding] please describe this image in details',
            '[grounding] describe this image as detailed as possible',
            '[grounding] summarize this image in details',
            '[grounding] give a thorough description of what you see in this image',
        ]

        with open(ann_path, 'r') as f:
            self.ann = json.load(f)

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        info = self.ann[index]

        image_file = info['folder_name']
        image_path = os.path.join(self.vis_root, image_file)
        grayscale_image = Image.open(image_path).convert("L")
        image = Image.new("RGB", grayscale_image.size)
        image.paste(grayscale_image)
        image = self.vis_processor(image)
    
        answer = info['grounded_caption']

        instruction = random.choice(self.instruction_pool)

        instruction = "[INST] <Img><ImageHere></Img> {} [/INST]".format(instruction)

        return add_default_structured_fields({
            "image": image,
            "instruction_input": instruction,
            "answer": answer,
            "image_id": info['folder_name'],
        }, modality="X-ray/CT/MRI", anatomy="multi-organ")


class SlakeVQADataset(Dataset):
    """SLAKE VQA dataset — reads ``img_name`` (not ``image_name``).

    JSON schema (per record)::
        {
            "img_id": 1,
            "img_name": "xmlab1/source.jpg",
            "question": "What modality is used to take this image?",
            "answer": "MRI",
            "q_lang": "en",
            "location": "Abdomen",
            "modality": "MRI",
            "answer_type": "OPEN",
            ...
        }
    """

    def __init__(self, vis_processor, text_processor, vis_root, ann_path):
        self.vis_root = vis_root
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.instruction_pool = ["[vqa] {}"]

        with open(ann_path, 'r') as f:
            self.ann = json.load(f)

    def process_image(self, img_name):
        image_path = os.path.join(self.vis_root, img_name)
        grayscale_image = Image.open(image_path).convert("L")
        image = Image.new("RGB", grayscale_image.size)
        image.paste(grayscale_image)
        return self.vis_processor(image)

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        info = self.ann[index]
        image = self.process_image(info['img_name'])
        instruction = self.text_processor(self.instruction_pool[0].format(info['question']))
        instruction = '[INST] <Img><ImageHere></Img> {} [/INST]'.format(instruction)

        answer = str(info['answer'])

        return add_default_structured_fields({
            "image": image,
            "instruction_input": instruction,
            "answer": answer,
            "image_id": info['img_name'],
        }, modality=info.get('modality', 'X-ray/CT/MRI'), anatomy=info.get('location', 'multi-organ'))


class evalSlakeVQADataset(Dataset):
    def __init__(self, loaded_data, vis_processor, root_path):
        self.loaded_data = loaded_data
        self.root_path = root_path
        self.vis_processor = vis_processor

    def __len__(self):
        return len(self.loaded_data)

    def __getitem__(self, idx):
        info = self.loaded_data[idx]
        image_file = info['img_name']
        image_path = os.path.join(self.root_path, image_file)
        grayscale_image = Image.open(image_path).convert("L")
        image = Image.new("RGB", grayscale_image.size)
        image.paste(grayscale_image)
        image = self.vis_processor(image)
        question = "[vqa] {}".format(info['question'])
        return image, question, image_file


class evalSLAKEDataset(Dataset):
    def __init__(self, loaded_data, vis_processor, root_path):
        self.loaded_data = loaded_data
        self.root_path = root_path
        self.vis_processor = vis_processor

    def __len__(self):
        return len(self.loaded_data)
    
    def __getitem__(self, idx):
        data = self.loaded_data[idx]
        img_id = data['folder_name']
        # sent = data['objects']
        image_path = os.path.join(self.root_path, img_id)
        grayscale_image = Image.open(image_path).convert("L")
        image = Image.new("RGB", grayscale_image.size)
        image.paste(grayscale_image)
        image = self.vis_processor(image)
        question = "[grounding] please describe this image in details"

        return image, question, img_id