"""
MiniGPT-Med Gradio Demo — based on official demo_gradio4-29.py.

Uses the Chat class for proper conversation management and bbox visualization.
Weights are auto-downloaded from ModelScope at startup.
"""

import os
import re
import random
import html
import sys
import subprocess
import logging
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Auto-install / pin package versions BEFORE importing anything that needs
#  them. IMPORTANT: version checks use importlib.metadata (reads installed
#  package metadata WITHOUT importing the module), so that after a pip
#  downgrade, the first real `import` loads the freshly installed version.
#  Importing the module first would cache the old version in sys.modules.
# ---------------------------------------------------------------------------
import importlib.metadata as _pkgmeta


def _pkg_version(name):
    try:
        return _pkgmeta.version(name)
    except _pkgmeta.PackageNotFoundError:
        return None


def _ensure_pkg(name):
    if _pkg_version(name) is None:
        logger.warning("Installing missing package: %s", name)
        subprocess.check_call([sys.executable, "-m", "pip", "install", name, "-q"])


for _pkg in ("webdataset", "decord", "wandb", "rich", "open_clip_torch", "opencv-python"):
    _ensure_pkg(_pkg)

# ---------------------------------------------------------------------------
#  Pinned-version matrix. All critical packages are checked as a GROUP and,
#  if any is out of range, they are reinstalled together in a SINGLE pip call
#  so that pip resolves one consistent dependency set (fixes HfFolder /
#  numpy.ma / cache_position issues caused by incremental --force-reinstall
#  where each package re-resolves and clobbers shared deps like
#  huggingface-hub).
# ---------------------------------------------------------------------------
_PIN = {
    "numpy":             ("1.26.4",  lambda v: v.startswith("1.26.")),
    "transformers":      ("4.36.2",  lambda v: v.startswith("4.36.")),
    "huggingface-hub":   ("0.26.2",  lambda v: v.startswith("0.26.")),
    "gradio":            ("4.44.1",  lambda v: v.startswith("4.44.")),
    "peft":              ("0.7.1",   lambda v: v.startswith("0.7.")),
    "accelerate":        ("0.28.0",  lambda v: v.startswith("0.28.")),
    "markupsafe":        ("2.0.1",   lambda v: v.startswith("2.")),
}


def _needs_pin(name, ok_fn):
    v = _pkg_version(name)
    if v is None:
        return True
    try:
        return not ok_fn(v)
    except Exception:
        return True


_pkgs_to_fix = [f"{n}=={v}" for n, (v, ok) in _PIN.items() if _needs_pin(n, ok)]
if _pkgs_to_fix:
    logger.warning("Pinning dependency set: %s", ", ".join(_pkgs_to_fix))
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install"] + _pkgs_to_fix + ["-q", "--force-reinstall"]
    )

# Pillow binary mismatch: Python code 12.x but _imaging C extension is 10.4.
# The server log reports "Core version: 10.4.0". pip uninstall/--force-reinstall
# leave stale 12.x .py files behind, so we DELETE every PIL directory on disk
# and reinstall a single clean pillow==10.4.0 (gradio 4.44.1 needs pillow<11).
import glob
import shutil
import site as _site

_pil_roots = []
for _sp in _site.getsitepackages():
    _p = os.path.join(_sp, "PIL")
    if os.path.isdir(_p):
        _pil_roots.append(_p)
# user site-packages too
_pil_roots += glob.glob(os.path.join(_site.getusersitepackages(), "PIL"))

_pil_ver = _pkg_version("pillow")
_pil_broken = _pil_ver != "10.4.0"
if not _pil_broken and _pil_roots:
    # Version says 10.4.0 but files may still be mixed; verify by listing
    # for a 12.x-only marker (ImageText.py importing _Ink).
    _marker = os.path.join(_pil_roots[0], "ImageText.py")
    if os.path.exists(_marker):
        _txt = Path(_marker).read_text(errors="ignore")
        if "_Ink" in _txt and "10.4" not in _txt:
            _pil_broken = True

if _pil_broken:
    logger.warning("Pillow %s broken/mixed — removing PIL dirs %s and reinstalling 10.4.0",
                   _pil_ver, _pil_roots)
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "pillow", "Pillow"])
    for _p in _pil_roots:
        shutil.rmtree(_p, ignore_errors=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "pillow==10.4.0", "-q", "--no-cache-dir"])

# Now (re)load the pinned versions cleanly.
import gradio as gr

# torchvision was broken by earlier pip operations (torchvision::nms / circular
# import).  Match torchvision version to the current torch version (PyTorch
# minor versions correspond 1:1 to torchvision minor versions).
import torch
_torch_minor = int(torch.__version__.split(".")[1])
# torch 2.X → torchvision 0.(15+X)  (2.3 → 0.18, 2.4 → 0.19, etc.)
_tv_expected_minor = 15 + _torch_minor
_tv_target = f"0.{_tv_expected_minor}.0"
_tv_ver = _pkg_version("torchvision")
# Always force-reinstall: version string may match but binary is corrupt
# from earlier pip operations (torchvision::nms / _meta_registrations).
logger.warning("Force-reinstalling torchvision %s→%s (torch %s) to fix binary",
               _tv_ver, _tv_target, torch.__version__)
subprocess.check_call([sys.executable, "-m", "pip", "install",
                       f"torchvision=={_tv_target}", "-q", "--force-reinstall"])

import cv2
import numpy as np
from PIL import Image
import torch
import torch.backends.cudnn as cudnn

from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Conversation, SeparatorStyle, Chat

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *


# ---------------------------------------------------------------------------
#  Config & paths
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).parent.absolute()
WEIGHTS_DIR = PROJECT_DIR / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True)

LLAMA_DIR = WEIGHTS_DIR / "llama-2-7b-chat-hf"
CKPT_PATH = WEIGHTS_DIR / "minigpt_med_pretrained.pth"
MODELSCOPE_CKPT = "ScholarChen20/miniGPT_Med"
MINIGPT_MED_FILENAME = "minigpt_med_pretrained.pth"


def download_minigpt_med():
    """Download MiniGPT-Med checkpoint from ModelScope Model repo."""
    if CKPT_PATH.exists():
        size_mb = CKPT_PATH.stat().st_size / 1e6
        logger.info("MiniGPT-Med checkpoint exists: %s (%.1f MB)", CKPT_PATH, size_mb)
        if size_mb < 5000:
            logger.warning("Checkpoint < 5GB — may be incomplete!")
        return True

    logger.info("Downloading MiniGPT-Med from ModelScope: %s", MODELSCOPE_CKPT)
    from modelscope.hub.snapshot_download import snapshot_download
    import shutil

    local_dir = snapshot_download(
        MODELSCOPE_CKPT, cache_dir=str(WEIGHTS_DIR), repo_type="model"
    )

    found = False
    for root, _dirs, files in os.walk(str(local_dir)):
        for f in files:
            if f == MINIGPT_MED_FILENAME:
                src = Path(root) / f
                src_mb = src.stat().st_size / 1e6
                logger.info("Found checkpoint: %s (%.1f MB)", src, src_mb)
                shutil.copy2(str(src), str(CKPT_PATH))
                found = True
                break
    if found:
        logger.info("Checkpoint ready: %s", CKPT_PATH)
        return True
    logger.error("Checkpoint NOT FOUND in %s", local_dir)
    return False


def download_llama():
    """Download LLaMA-2-7B-chat from ModelScope if not present."""
    if LLAMA_DIR.exists() and any(LLAMA_DIR.iterdir()):
        logger.info("LLaMA-2-7B already exists: %s", LLAMA_DIR)
        return True

    logger.info("Downloading LLaMA-2-7B-chat-hf from ModelScope...")
    from modelscope.hub.snapshot_download import snapshot_download
    snapshot_download(
        "shakechen/Llama-2-7b-chat-hf",
        cache_dir=str(WEIGHTS_DIR),
        local_dir=str(LLAMA_DIR),
        repo_type="model",
    )
    logger.info("LLaMA-2-7B ready: %s", LLAMA_DIR)
    return True


# ---------------------------------------------------------------------------
#  Model initialization (following official demo pattern)
# ---------------------------------------------------------------------------
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
cudnn.benchmark = False
cudnn.deterministic = True

# Ensure weights are available
download_minigpt_med()
download_llama()

# Build config WITHOUT the training-oriented Config class (avoids needing
# dataset builder yamls like cc_sbu/align.yaml). We only need model config
# and the image processor config, both defined in minigpt_v2.yaml.
from omegaconf import OmegaConf

_model_cfg_path = PROJECT_DIR / "minigpt4" / "configs" / "models" / "minigpt_v2.yaml"
_cfg = OmegaConf.load(str(_model_cfg_path))
model_config = _cfg.model

# Override paths to point to our downloaded weights
model_config.llama_model = str(LLAMA_DIR)
model_config.ckpt = str(CKPT_PATH)
model_config.low_resource = False
model_config.device_8bit = 0

device = "cuda:0" if torch.cuda.is_available() else "cpu"

logger.info("Loading model (arch=%s)...", model_config.arch)
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to(device)
model = model.eval()

# Image processor — build from the preprocess section (eval split)
vis_processor_cfg = _cfg.preprocess.vis_processor.eval
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

bounding_box_size = 100

# Official conversation template — note the <s> BOS prefix!
CONV_VISION = Conversation(
    system="",
    roles=(r"<s>[INST] ", r" [/INST]"),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="",
)

chat = Chat(model, vis_processor, device=device)
logger.info("Model loaded and Chat initialized.")


# ---------------------------------------------------------------------------
#  Bbox visualization (from official demo)
# ---------------------------------------------------------------------------
colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (210, 210, 0),
    (255, 0, 255), (0, 255, 255), (114, 128, 250), (0, 165, 255),
    (0, 128, 0), (144, 238, 144), (238, 238, 175), (255, 191, 0),
    (0, 128, 0), (226, 43, 138), (255, 0, 255), (0, 215, 255),
]

color_map = {
    f"{color_id}": f"#{hex(color[2])[2:].zfill(2)}{hex(color[1])[2:].zfill(2)}{hex(color[0])[2:].zfill(2)}"
    for color_id, color in enumerate(colors)
}
used_colors = colors


def extract_substrings(string):
    index = string.rfind('}')
    if index != -1:
        string = string[:index + 1]
    pattern = r'<p>(.*?)\}(?!<)'
    matches = re.findall(pattern, string)
    return [match for match in matches]


def is_overlapping(rect1, rect2):
    x1, y1, x2, y2 = rect1
    x3, y3, x4, y4 = rect2
    return not (x2 < x3 or x1 > x4 or y2 < y3 or y1 > y4)


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
    return intersection_area / union_area


def save_tmp_img(visual_img):
    tmp_dir = "/tmp/gradio"
    os.makedirs(tmp_dir, exist_ok=True)
    file_name = "".join([str(random.randint(0, 9)) for _ in range(5)]) + ".jpg"
    file_path = os.path.join(tmp_dir, file_name)
    visual_img.save(file_path)
    return file_path


def mask2bbox(mask):
    if mask is None:
        return ''
    mask = mask.resize([100, 100], resample=Image.NEAREST)
    mask = np.array(mask)[:, :, 0]
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if rows.sum():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        bbox = '{{<{}><{}><{}><{}>}}'.format(cmin, rmin, cmax, rmax)
    else:
        bbox = ''
    return bbox


def escape_markdown(text):
    for char in ['<', '>']:
        text = text.replace(char, '\\' + char)
    return text


def reverse_escape(text):
    for char in ['\\<', '\\>']:
        text = text.replace(char, char[1:])
    return text


def visualize_all_bbox_together(image, generation):
    if image is None:
        return None, ''

    generation = html.unescape(generation)
    print('gen begin', generation)
    image_width, image_height = image.size
    image = image.resize([500, int(500 / image_width * image_height)])
    image_width, image_height = image.size

    string_list = extract_substrings(generation)
    if string_list:
        mode = 'all'
        entities = defaultdict(list)
        i = 0
        j = 0
        for string in string_list:
            try:
                obj, string = string.split('</p>')
            except ValueError:
                print('wrong string: ', string)
                continue
            bbox_list = string.split('<delim>')
            flag = False
            for bbox_string in bbox_list:
                integers = re.findall(r'-?\d+', bbox_string)
                if len(integers) == 4:
                    x0, y0, x1, y1 = int(integers[0]), int(integers[1]), int(integers[2]), int(integers[3])
                    left = x0 / bounding_box_size * image_width
                    bottom = y0 / bounding_box_size * image_height
                    right = x1 / bounding_box_size * image_width
                    top = y1 / bounding_box_size * image_height
                    entities[obj].append([left, bottom, right, top])
                    j += 1
                    flag = True
            if flag:
                i += 1
    else:
        integers = re.findall(r'-?\d+', generation)
        if len(integers) == 4:
            mode = 'single'
            entities = list()
            x0, y0, x1, y1 = int(integers[0]), int(integers[1]), int(integers[2]), int(integers[3])
            left = x0 / bounding_box_size * image_width
            bottom = y0 / bounding_box_size * image_height
            right = x1 / bounding_box_size * image_width
            top = y1 / bounding_box_size * image_height
            entities.append([left, bottom, right, top])
        else:
            return None, ''

    if len(entities) == 0:
        return None, ''

    if isinstance(image, Image.Image):
        image_h = image.height
        image_w = image.width
        image = np.array(image)
    elif isinstance(image, str):
        if os.path.exists(image):
            pil_img = Image.open(image).convert("RGB")
            image = np.array(pil_img)[:, :, [2, 1, 0]]
            image_h = pil_img.height
            image_w = pil_img.width
        else:
            raise ValueError(f"invaild image path, {image}")
    elif isinstance(image, torch.Tensor):
        # (C, H, W) tensor → (H, W, C) numpy, denormalized to [0,1]
        image_tensor = image.cpu().float()
        reverse_norm_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])[:, None, None]
        reverse_norm_std = torch.tensor([0.26862954, 0.26130258, 0.27577711])[:, None, None]
        image_tensor = image_tensor * reverse_norm_std + reverse_norm_mean
        img_np = image_tensor.permute(1, 2, 0).clamp(0.0, 1.0).mul(255).byte().numpy()
        pil_img = Image.fromarray(img_np)
        image_h = pil_img.height
        image_w = pil_img.width
        image = np.array(pil_img)[:, :, [2, 1, 0]]
    else:
        raise ValueError(f"invaild image format, {type(image)} for {image}")

    indices = list(range(len(entities)))
    new_image = image.copy()
    previous_bboxes = []
    text_size = 0.5
    text_line = 1
    box_line = 2
    (c_width, text_height), _ = cv2.getTextSize("F", cv2.FONT_HERSHEY_COMPLEX, text_size, text_line)
    base_height = int(text_height * 0.675)
    text_offset_original = text_height - base_height
    text_spaces = 2
    used_colors = colors
    color_id = -1

    for entity_idx, entity_name in enumerate(entities):
        if mode == 'single' or mode == 'identify':
            bboxes = entity_name
            bboxes = [bboxes]
        else:
            bboxes = entities[entity_name]
        color_id += 1
        for bbox_id, (x1_norm, y1_norm, x2_norm, y2_norm) in enumerate(bboxes):
            skip_flag = False
            orig_x1, orig_y1, orig_x2, orig_y2 = int(x1_norm), int(y1_norm), int(x2_norm), int(y2_norm)
            color = used_colors[entity_idx % len(used_colors)]
            new_image = cv2.rectangle(new_image, (orig_x1, orig_y1), (orig_x2, orig_y2), color, box_line)

            if mode == 'all':
                l_o, r_o = box_line // 2 + box_line % 2, box_line // 2 + box_line % 2 + 1
                x1 = orig_x1 - l_o
                y1 = orig_y1 - l_o
                if y1 < text_height + text_offset_original + 2 * text_spaces:
                    y1 = orig_y1 + r_o + text_height + text_offset_original + 2 * text_spaces
                    x1 = orig_x1 + r_o
                (text_width, text_height), _ = cv2.getTextSize(f"  {entity_name}", cv2.FONT_HERSHEY_COMPLEX, text_size, text_line)
                text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2 = x1, y1 - (text_height + text_offset_original + 2 * text_spaces), x1 + text_width, y1

                for prev_bbox in previous_bboxes:
                    if computeIoU((text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2), prev_bbox['bbox']) > 0.95 and prev_bbox['phrase'] == entity_name:
                        skip_flag = True
                        break
                    while is_overlapping((text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2), prev_bbox['bbox']):
                        text_bg_y1 += (text_height + text_offset_original + 2 * text_spaces)
                        text_bg_y2 += (text_height + text_offset_original + 2 * text_spaces)
                        y1 += (text_height + text_offset_original + 2 * text_spaces)
                        if text_bg_y2 >= image_h:
                            text_bg_y1 = max(0, image_h - (text_height + text_offset_original + 2 * text_spaces))
                            text_bg_y2 = image_h
                            y1 = image_h
                            break
                if not skip_flag:
                    alpha = 0.5
                    for i in range(text_bg_y1, text_bg_y2):
                        for j in range(text_bg_x1, text_bg_x2):
                            if i < image_h and j < image_w:
                                if j < text_bg_x1 + 1.35 * c_width:
                                    bg_color = color
                                else:
                                    bg_color = [255, 255, 255]
                                new_image[i, j] = (alpha * new_image[i, j] + (1 - alpha) * np.array(bg_color)).astype(np.uint8)
                    cv2.putText(new_image, f"  {entity_name}", (x1, y1 - text_offset_original - 1 * text_spaces), cv2.FONT_HERSHEY_COMPLEX, text_size, (0, 0, 0), text_line, cv2.LINE_AA)
                    previous_bboxes.append({'bbox': (text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2), 'phrase': entity_name})

    if mode == 'all':
        def color_iterator(colors):
            while True:
                for color in colors:
                    yield color
        color_gen = color_iterator(colors)

        def colored_phrases(match):
            phrase = match.group(1)
            color = next(color_gen)
            return f'<span style="color:rgb{color}">{phrase}</span>'

        generation = re.sub(r'{<\d+><\d+><\d+><\d+>}|<delim>', '', generation)
        generation_colored = re.sub(r'<p>(.*?)</p>', colored_phrases, generation)
    else:
        generation_colored = ''

    pil_image = Image.fromarray(new_image)
    return pil_image, generation_colored


# ---------------------------------------------------------------------------
#  Gradio handlers (from official demo)
# ---------------------------------------------------------------------------
def gradio_reset(chat_state, img_list):
    if chat_state is not None:
        chat_state.messages = []
    if img_list is not None:
        img_list = []
    # Return new empty state objects (avoids gr.update() API differences)
    return None, None, "", chat_state, img_list


def image_upload_trigger(upload_flag, replace_flag, img_list):
    upload_flag = 1
    if img_list:
        replace_flag = 1
    return upload_flag, replace_flag


def gradio_ask(user_message, chatbot, chat_state, gr_img, img_list, upload_flag, replace_flag):
    if len(user_message) == 0:
        text_box_show = 'Input should not be empty!'
    else:
        text_box_show = ''

    if isinstance(gr_img, dict):
        gr_img, mask = gr_img['image'], gr_img['mask']
    else:
        mask = None

    if '[identify]' in user_message:
        integers = re.findall(r'-?\d+', user_message)
        if len(integers) != 4:
            bbox = mask2bbox(mask)
            user_message = user_message + bbox

    if chat_state is None:
        chat_state = CONV_VISION.copy()

    if upload_flag:
        if replace_flag:
            chat_state = CONV_VISION.copy()
            replace_flag = 0
            chatbot = []
        img_list = []
        llm_message = chat.upload_img(gr_img, chat_state, img_list)
        upload_flag = 0

    chat.ask(user_message, chat_state)
    chatbot = chatbot + [[user_message, None]]

    if '[identify]' in user_message:
        visual_img, _ = visualize_all_bbox_together(gr_img, user_message)
        if visual_img is not None:
            file_path = save_tmp_img(visual_img)
            chatbot = chatbot + [[(file_path,), None]]

    return text_box_show, chatbot, chat_state, img_list, upload_flag, replace_flag


def gradio_stream_answer(chatbot, chat_state, img_list, temperature):
    if len(img_list) > 0:
        if not isinstance(img_list[0], torch.Tensor):
            chat.encode_img(img_list)
    streamer = chat.stream_answer(conv=chat_state, img_list=img_list, temperature=temperature, max_new_tokens=500, max_length=2000)
    output = ''
    for new_output in streamer:
        escapped = escape_markdown(new_output)
        output += escapped
        chatbot[-1][1] = output
        yield chatbot, chat_state
    chat_state.messages[-1][1] = '</s>'
    return chatbot, chat_state


def gradio_visualize(chatbot, gr_img):
    if isinstance(gr_img, dict):
        gr_img, mask = gr_img['image'], gr_img['mask']
    unescaped = reverse_escape(chatbot[-1][1])
    visual_img, generation_color = visualize_all_bbox_together(gr_img, unescaped)
    if visual_img is not None:
        if len(generation_color):
            chatbot[-1][1] = generation_color
        file_path = save_tmp_img(visual_img)
        chatbot = chatbot + [[None, (file_path,)]]
    return chatbot


def gradio_taskselect(idx):
    prompt_list = [
        '',
        '[grounding] describe this image in detail',
        '[refer] ',
        '[detection] ',
        '[identify] what is this ',
        '[vqa] ',
        '[caption] Could you describe the contents of this image for me?',
    ]
    instruct_list = [
        '**Hint:** Type in whatever you want',
        '**Hint:** Send the command to generate a grounded image description',
        '**Hint:** Type in a phrase about an object in the image and send the command',
        '**Hint:** Type in a caption or phrase, and see object locations in the image',
        '**Hint:** Draw a bounding box on the uploaded image then send the command. Click the "clear" button on the top right of the image before redraw',
        '**Hint:** Send a question to get a short answer',
        '**Hint:** Send to generate a medical report',
    ]
    return prompt_list[idx], instruct_list[idx]


# ---------------------------------------------------------------------------
#  UI
# ---------------------------------------------------------------------------
title = """<h1 align="center">MiniGPT-Med</h1>"""

introduction = '''
For Abilities Involving Visual Grounding:
1. **No Tag**: Input whatever you want and CLICK **Send** without any tagging
2. **Grounding**: CLICK **Send** to generate a grounded image description.
3. **Refer**: Input a referring object and CLICK **Send**.
4. **Detection**: Write a caption or phrase, and CLICK **Send**.
5. **Identify**: Draw the bounding box on the uploaded image window and CLICK **Send** to generate the bounding box. (CLICK "clear" button before re-drawing next time).
6. **VQA**: Input a visual question and CLICK **Send**.
7. **Report**: CLICK **Send** to generate a medical report.

You can also simply chat in free form!
'''

with gr.Blocks() as demo:
    gr.Markdown(title)

    with gr.Row():
        with gr.Column(scale=1):
            image = gr.Image(type="pil")
            temperature = gr.Slider(
                minimum=0.1, maximum=1.5, value=0.6, step=0.1,
                interactive=True, label="Temperature",
            )
            clear = gr.Button("Restart")
            gr.Markdown(introduction)

        with gr.Column(scale=2):
            chat_state = gr.State(value=None)
            img_list = gr.State(value=[])
            chatbot = gr.Chatbot(label='MiniGPT-Med')

            dataset = gr.Dataset(
                components=[gr.Textbox(visible=False)],
                samples=[['No Tag'], ['Grounding'], ['Refer'], ['Detection'], ['Identify'], ['VQA'], ['Report']],
                type="index",
                label='Task Shortcuts',
            )
            task_inst = gr.Markdown('**Hint:** Upload your image and chat')
            with gr.Row():
                text_input = gr.Textbox(
                    placeholder='Upload your image and chat', interactive=True,
                    show_label=False, container=False, scale=8,
                )
                send = gr.Button("Send", variant='primary', scale=1)

    upload_flag = gr.State(value=0)
    replace_flag = gr.State(value=0)
    image.upload(image_upload_trigger, [upload_flag, replace_flag, img_list], [upload_flag, replace_flag])

    dataset.click(
        gradio_taskselect,
        inputs=[dataset],
        outputs=[text_input, task_inst],
        show_progress="hidden",
        postprocess=False,
        queue=False,
    )

    text_input.submit(
        gradio_ask,
        [text_input, chatbot, chat_state, image, img_list, upload_flag, replace_flag],
        [text_input, chatbot, chat_state, img_list, upload_flag, replace_flag],
        queue=False,
    ).success(
        gradio_stream_answer,
        [chatbot, chat_state, img_list, temperature],
        [chatbot, chat_state],
    ).success(
        gradio_visualize,
        [chatbot, image],
        [chatbot],
        queue=False,
    )

    send.click(
        gradio_ask,
        [text_input, chatbot, chat_state, image, img_list, upload_flag, replace_flag],
        [text_input, chatbot, chat_state, img_list, upload_flag, replace_flag],
        queue=False,
    ).success(
        gradio_stream_answer,
        [chatbot, chat_state, img_list, temperature],
        [chatbot, chat_state],
    ).success(
        gradio_visualize,
        [chatbot, image],
        [chatbot],
        queue=False,
    )

    clear.click(gradio_reset, [chat_state, img_list], [chatbot, image, text_input, chat_state, img_list], queue=False)

if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860)
