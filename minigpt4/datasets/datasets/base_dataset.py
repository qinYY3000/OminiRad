"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
from typing import Iterable

from torch.utils.data import Dataset, ConcatDataset
from torch.utils.data.dataloader import default_collate


class BaseDataset(Dataset):
    def __init__(
        self, vis_processor=None, text_processor=None, vis_root=None, ann_paths=[]
    ):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.vis_root = vis_root

        self.annotation = []
        # print("ann paths", ann_paths)
        for ann_path in ann_paths:
            # print("ann_path", ann_path)
            ann = json.load(open(ann_path, "r"))
            if isinstance(ann, dict):
                self.annotation.extend(json.load(open(ann_path, "r"))['annotations'])
                # self.annotation.extend(json.load(open(ann_path, "r")))
            else:
                self.annotation.extend(json.load(open(ann_path, "r")))
    
        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self._add_instance_ids()

    def __len__(self):
        return len(self.annotation)

    def collater(self, samples):
        return default_collate(samples)

    def set_processors(self, vis_processor, text_processor):
        self.vis_processor = vis_processor
        self.text_processor = text_processor

    def _add_instance_ids(self, key="instance_id"):
        for idx, ann in enumerate(self.annotation):
            ann[key] = str(idx)


class ConcatDataset(ConcatDataset):
    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super().__init__(datasets)

    def collater(self, samples):
        """Collate samples that may come from different underlying datasets.

        Uses the **union** of all keys (not intersection) so that
        structured-supervision fields from ``UnifiedUSDataset`` are preserved
        even when mixed with legacy datasets.  Missing keys are filled with
        ``None`` per sample before dispatching to the first dataset's
        ``collater`` (or ``default_collate`` if none).
        """
        # Collect the union of all keys across samples
        all_keys: set = set()
        for s in samples:
            all_keys.update(s.keys())

        # Fill missing keys with None so every sample has the same schema
        unified_samples = []
        for s in samples:
            unified = dict(s)
            for key in all_keys:
                if key not in unified:
                    unified[key] = None
            unified_samples.append(unified)

        # Dispatch to the first dataset's collater, or fall back to default_collate
        if hasattr(self.datasets[0], "collater"):
            return self.datasets[0].collater(unified_samples)
        return default_collate(unified_samples)
