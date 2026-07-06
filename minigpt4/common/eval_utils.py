import argparse
import numpy as np
from nltk.translate.bleu_score import sentence_bleu

from minigpt4.common.registry import registry
from minigpt4.common.config import Config

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *



def eval_parser():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--name", type=str, default='A2', help="evaluation name")
    parser.add_argument("--ckpt", type=str, help="path to configuration file.")
    parser.add_argument("--eval_opt", type=str, default='all', help="path to configuration file.")
    parser.add_argument("--max_new_tokens", type=int, default=10, help="max number of generated tokens")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lora_r", type=int, default=64, help="lora rank of the model")
    parser.add_argument("--lora_alpha", type=int, default=16, help="lora alpha")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )
    return parser


def prepare_texts(texts, conv_temp):
    convs = [conv_temp.copy() for _ in range(len(texts))]
    [conv.append_message(
        conv.roles[0], '<Img><ImageHere></Img> {}'.format(text)) for conv, text in zip(convs, texts)]
    [conv.append_message(conv.roles[1], None) for conv in convs]
    texts = [conv.get_prompt() for conv in convs]
    return texts


def init_model(args):
    print('Initialization Model')
    cfg = Config(args)
    # cfg.model_cfg.ckpt = args.ckpt
    # cfg.model_cfg.lora_r = args.lora_r
    # cfg.model_cfg.lora_alpha = args.lora_alpha

    model_config = cfg.model_cfg
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:0')

#     import pudb; pudb.set_trace()
    key = list(cfg.datasets_cfg.keys())[0]
    vis_processor_cfg = cfg.datasets_cfg.get(key).vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    print('Initialization Finished')
    return model, vis_processor

def computeIoU(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2
    intersection_x1 = max(x1, x3)
    intersection_y1 = max(y1, y3)
    intersection_x2 = min(x2, x4)
    intersection_y2 = min(y2, y4)
    intersection_area = max(0, intersection_x2 - intersection_x1 + 1) * max(0, intersection_y2 - intersection_y1 + 1)
    bbox1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    bbox2_area = (x4 - x3 + 1) * (y4 - y3 + 1)
    union_area = bbox1_area + bbox2_area - intersection_area
    iou = intersection_area / union_area
    return iou


def compute_giou(bbox1, bbox2):
    """Generalized IoU for bounding boxes.

    GIoU = IoU - |enclosing_area - union_area| / |enclosing_area|
    Returns value in [-1, 1].
    """
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2
    ix1, iy1 = max(x1, x3), max(y1, y3)
    ix2, iy2 = min(x2, x4), min(y2, y4)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area1 = max(0, x2 - x1) * max(0, y2 - y1)
    area2 = max(0, x4 - x3) * max(0, y4 - y3)
    union = area1 + area2 - intersection
    if union == 0:
        return 0.0
    iou = intersection / union
    # Enclosing box
    ex1, ey1 = min(x1, x3), min(y1, y3)
    ex2, ey2 = max(x2, x4), max(y2, y4)
    enclosing = max(0, ex2 - ex1) * max(0, ey2 - ey1)
    giou = iou - (enclosing - union) / enclosing if enclosing > 0 else iou
    return giou


def dice_score(pred_mask, gt_mask, threshold=0.5):
    """Dice coefficient for binary masks.

    Parameters
    ----------
    pred_mask : numpy array or torch tensor — predicted mask (probabilities or logits)
    gt_mask : numpy array or torch tensor — ground truth binary mask
    threshold : float — threshold to binarize pred_mask

    Returns
    -------
    float in [0, 1]
    """
    import numpy as np
    if hasattr(pred_mask, 'cpu'):
        pred_mask = pred_mask.cpu().numpy()
    if hasattr(gt_mask, 'cpu'):
        gt_mask = gt_mask.cpu().numpy()
    pred_bin = (pred_mask > threshold).astype(np.float32)
    gt_bin = (gt_mask > 0.5).astype(np.float32)
    intersection = (pred_bin * gt_bin).sum()
    return (2.0 * intersection) / (pred_bin.sum() + gt_bin.sum() + 1e-8)


def mask_iou(pred_mask, gt_mask, threshold=0.5):
    """Mask IoU for binary masks.

    Returns
    -------
    float in [0, 1]
    """
    import numpy as np
    if hasattr(pred_mask, 'cpu'):
        pred_mask = pred_mask.cpu().numpy()
    if hasattr(gt_mask, 'cpu'):
        gt_mask = gt_mask.cpu().numpy()
    pred_bin = (pred_mask > threshold).astype(np.float32)
    gt_bin = (gt_mask > 0.5).astype(np.float32)
    intersection = (pred_bin * gt_bin).sum()
    union = pred_bin.sum() + gt_bin.sum() - intersection
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union

