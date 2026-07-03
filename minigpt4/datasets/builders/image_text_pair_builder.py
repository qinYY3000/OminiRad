import os
import logging
import warnings

from minigpt4.common.registry import registry
from minigpt4.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from minigpt4.datasets.datasets.cc_sbu_dataset import CCSBUDataset, CCSBUAlignDataset
from minigpt4.datasets.datasets.radvqa_dataset import RadVQADataset
from minigpt4.datasets.datasets.rsna_dataset import RSNADataset,ReferRSNADataset,IdentifyRSNADataset
from minigpt4.datasets.datasets.nlst_dataset import NlstDataset,ReferNLSTDataset,IdentifyNLSTDataset
from minigpt4.datasets.datasets.indiana_dataset import IndianaCXRDataset

from minigpt4.datasets.datasets.SLAKE_dataset import GroundingSLAKEDatase, SlakeVQADataset
from minigpt4.datasets.datasets.unified_us_dataset import UnifiedUSDataset, GroupBreastUSDataset, GroupThyroidUSDataset


def _resolve_path(path: str) -> str:
    """Resolve a possibly-relative path against the project repo_root.

    - Absolute paths are returned as-is.
    - Relative paths (e.g. ``data/Chest X/images``) are joined with the
      registered ``repo_root`` so that training works regardless of the
      current working directory.
    - Environment variables (``$HOME``, ``${OMNIRAD_DATA_ROOT}`` ...) are
      expanded first.
    """
    if not path:
        return path
    path = os.path.expandvars(path)
    if os.path.isabs(path):
        return path
    try:
        repo_root = registry.get_path("repo_root")
    except (KeyError, AssertionError):
        repo_root = os.getcwd()
    return os.path.join(repo_root, path)


class _PathResolvingBuilder(BaseDatasetBuilder):
    """Base builder that resolves relative image_path / ann_path against repo_root.

    Supports an optional ``val_ann_path`` in the dataset config's ``build_info``.
    When present, ``build_datasets()`` returns both ``train`` and ``val`` splits
    so the training loop can do best-checkpoint selection.
    """

    def _build_train_dataset(self):
        self.build_processors()
        build_info = self.config.build_info
        ann_path = _resolve_path(build_info.ann_path)
        vis_root = _resolve_path(build_info.image_path)
        return self.train_dataset_cls(
            vis_processor=self.vis_processors['train'],
            text_processor=self.text_processors['train'],
            ann_path=ann_path,
            vis_root=vis_root,
        )

    def _build_val_dataset(self):
        """Build a validation dataset if ``val_ann_path`` is configured."""
        self.build_processors()
        build_info = self.config.build_info
        val_ann = getattr(build_info, "val_ann_path", None) or build_info.get("val_ann_path", None)
        if not val_ann:
            return None
        vis_root = _resolve_path(build_info.image_path)
        return self.train_dataset_cls(
            vis_processor=self.vis_processors['train'],
            text_processor=self.text_processors['train'],
            ann_path=_resolve_path(val_ann),
            vis_root=vis_root,
        )

    def build_datasets(self):
        """Default: return ``train`` split and (optionally) ``val`` split.

        Subclasses that don't need val (e.g. CCSBUAlignBuilder) override this.
        """
        datasets = dict()
        datasets['train'] = self._build_train_dataset()
        val_ds = self._build_val_dataset()
        if val_ds is not None:
            datasets['val'] = val_ds
        return datasets


@registry.register_builder("cc_sbu_align")
class CCSBUAlignBuilder(BaseDatasetBuilder):
    train_dataset_cls = CCSBUAlignDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/cc_sbu/align.yaml",
    }

    def build_datasets(self):
        logging.info("Building datasets...")
        self.build_processors()
        build_info = self.config.build_info
        storage_path = _resolve_path(build_info.storage)

        datasets = dict()
        if not os.path.exists(storage_path):
            warnings.warn("storage path {} does not exist.".format(storage_path))

        datasets['train'] = self.train_dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'filter_cap.json')],
            vis_root=os.path.join(storage_path, 'image'),
        )
        return datasets

@registry.register_builder("indiana_cxr")
class IndianaCxrBuilder(_PathResolvingBuilder):
    """Indiana University Chest-X-ray (Open-i) builder.

    Inherits ``build_datasets()`` from ``_PathResolvingBuilder`` which returns
    both ``train`` and ``val`` splits (when ``val_ann_path`` is set in the YAML).
    """
    train_dataset_cls = IndianaCXRDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/indiana_cxr/indiana_cxr.yaml",
    }


@registry.register_builder("radvqa")
class RadVQABuilder(_PathResolvingBuilder):
    train_dataset_cls = RadVQADataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/radvqa/radvqa.yaml",
    }


@registry.register_builder("rsna")
class RSNABuilder(_PathResolvingBuilder):
    train_dataset_cls = RSNADataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/rsna/rsna.yaml",
    }


@registry.register_builder("refer_rsna")
class ReferRSNABuilder(_PathResolvingBuilder):
    train_dataset_cls = ReferRSNADataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/refer_rsna/refer_rsna.yaml",
    }


@registry.register_builder("identify_rsna")
class IdentifyRSNABuilder(_PathResolvingBuilder):
    train_dataset_cls = IdentifyRSNADataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/identify_rsna/identify_rsna.yaml",
    }


@registry.register_builder("nlst")
class NlstBuilder(_PathResolvingBuilder):
    train_dataset_cls = NlstDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/nlst/nlst.yaml",
    }


@registry.register_builder("refer_nlst")
class ReferNLSTBuilder(_PathResolvingBuilder):
    train_dataset_cls = ReferNLSTDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/refer_nlst/refer_nlst.yaml",
    }


@registry.register_builder("identify_nlst")
class IdentifyNLSTBuilder(_PathResolvingBuilder):
    train_dataset_cls = IdentifyNLSTDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/identify_nlst/identify_nlst.yaml",
    }


@registry.register_builder("group_breast_us")
class GroupBreastUSBuilder(_PathResolvingBuilder):
    train_dataset_cls = GroupBreastUSDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/group_breast_us/group_breast_us.yaml",
    }


@registry.register_builder("group_thyroid_us")
class GroupThyroidUSBuilder(_PathResolvingBuilder):
    train_dataset_cls = GroupThyroidUSDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/group_thyroid_us/group_thyroid_us.yaml",
    }


@registry.register_builder("grounding_SLAKE")
class GroundingSLAKEBuilder(_PathResolvingBuilder):
    train_dataset_cls = GroundingSLAKEDatase
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/grounding_SLAKE/grounding_SLAKE.yaml",
    }


@registry.register_builder("slake_vqa")
class SlakeVQABuilder(_PathResolvingBuilder):
    train_dataset_cls = SlakeVQADataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/slake_vqa/slake_vqa.yaml",
    }
