"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import sys

from omegaconf import OmegaConf

from minigpt4.common.registry import registry

# Datasets/tasks are only needed for training/eval; skip gracefully for inference
try:
    from minigpt4.datasets.builders import *
except Exception as e:  # ImportError OR RuntimeError (broken torchvision etc.)
    import logging
    logging.warning("Skipping dataset builders (not needed for inference): %s", e)

from minigpt4.models import *

try:
    from minigpt4.processors import *
except Exception as e:
    import logging
    logging.warning("Skipping processors (not needed for inference): %s", e)

try:
    from minigpt4.tasks import *
except Exception as e:
    import logging
    logging.warning("Skipping tasks (not needed for inference): %s", e)


root_dir = os.path.dirname(os.path.abspath(__file__))
default_cfg = OmegaConf.load(os.path.join(root_dir, "configs/default.yaml"))

registry.register_path("library_root", root_dir)
repo_root = os.path.join(root_dir, "..")
registry.register_path("repo_root", repo_root)
cache_root = os.path.join(repo_root, default_cfg.env.cache_root)
registry.register_path("cache_root", cache_root)

registry.register("MAX_INT", sys.maxsize)
registry.register("SPLIT_NAMES", ["train", "val", "test"])
