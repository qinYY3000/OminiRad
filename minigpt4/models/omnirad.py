from __future__ import annotations

import logging
import os
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from minigpt4.common.registry import registry
from minigpt4.models.minigpt_v2 import MiniGPTv2
from transformers import StoppingCriteriaList


def _to_plain_dict(cfg: Any) -> dict:
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg
    return OmegaConf.to_container(cfg, resolve=True)


# ============================================================================
# SAM integration: real dense encoder + mask decoder
# ============================================================================

try:
    from segment_anything import sam_model_registry
    _SAM_AVAILABLE = True
except ImportError:
    _SAM_AVAILABLE = False


SAM_MODEL_ALIASES = {
    "vit_h": "vit_h",
    "vit_l": "vit_l",
    "vit_b": "vit_b",
    "sam_vit_h": "vit_h",
    "sam_vit_l": "vit_l",
    "sam_vit_b": "vit_b",
    "medsam_vit_b": "vit_b",
}


def _resolve_sam_model_type(name: str | None, default: str = "vit_h") -> str:
    requested = (name or default).lower()
    fallback = SAM_MODEL_ALIASES.get(default.lower(), default.lower())
    resolved = SAM_MODEL_ALIASES.get(requested, requested)

    if not _SAM_AVAILABLE:
        return resolved

    if resolved in sam_model_registry:
        return resolved

    if fallback in sam_model_registry:
        logging.warning(
            "Unknown SAM backbone '%s'; falling back to '%s'.",
            requested,
            fallback,
        )
        return fallback

    return next(iter(sam_model_registry))


class SamDenseEncoder(nn.Module):

    """Wraps SAM ViT-H image encoder for dense spatial features.

    Input:  ``(B, 3, H, W)`` image tensor (will be resized to 1024 internally)
    Output: ``(B, 256, 64, 64)`` dense image embedding
    """

    def __init__(self, name: str = "sam_vit_h", weights: str | None = None,
                 freeze: bool = True, no_grad_forward: bool = True):
        super().__init__()
        self.name = name
        self.freeze = freeze
        self.no_grad_forward = no_grad_forward
        self.weights = weights

        sam_type = _resolve_sam_model_type(name, default="vit_h")
        sam = sam_model_registry[sam_type](checkpoint=weights or None)


        self.image_encoder = sam.image_encoder
        self.pixel_mean = nn.Parameter(
            torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1), requires_grad=False
        )
        self.pixel_std = nn.Parameter(
            torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1), requires_grad=False
        )
        self.target_size = 1024

        if freeze:
            for param in self.image_encoder.parameters():
                param.requires_grad = False
            self.image_encoder.eval()

        logging.info("SamDenseEncoder initialized: %s (freeze=%s)", sam_type, freeze)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image to dense feature map.

        Parameters
        ----------
        image : ``(B, 3, H, W)`` — image tensor in [0, 1] or [0, 255] range

        Returns
        -------
        ``(B, 256, 64, 64)`` image embedding
        """
        # Resize to 1024x1024
        img = F.interpolate(image, size=(self.target_size, self.target_size),
                            mode="bilinear", align_corners=False)
        # Normalize (SAM expects ImageNet normalization)
        if img.max() <= 1.0:
            img = img * 255.0
        img = (img - self.pixel_mean) / self.pixel_std

        if self.no_grad_forward:
            with torch.no_grad():
                return self.image_encoder(img)
        return self.image_encoder(img)


class SamMaskDecoder(nn.Module):
    """Wraps SAM's mask decoder for LLM-token-driven mask prediction.

    Takes the SEG token embedding (projected to 256-d) as sparse prompt,
    plus the dense image embedding, and produces a binary mask.
    """

    def __init__(
        self,
        name: str = "sam_vit_h",
        weights: str | None = None,
        predict_uncertainty: bool = True,
    ):
        super().__init__()
        sam_type = _resolve_sam_model_type(name, default="vit_h")
        sam = sam_model_registry[sam_type](checkpoint=weights or None)


        self.mask_decoder = sam.mask_decoder
        self.predict_uncertainty = predict_uncertainty

        # Uncertainty head (parallel to mask head)
        if predict_uncertainty:
            self.uncertainty_head = nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(128, 1, kernel_size=1),
            )

        logging.info("SamMaskDecoder initialized (uncertainty=%s)", predict_uncertainty)

    def forward(
        self,
        sparse_prompt: torch.Tensor,   # (B, 1, 256) — SEG token embedding
        image_embedding: torch.Tensor,  # (B, 256, 64, 64) — from SamDenseEncoder
    ) -> dict[str, torch.Tensor]:
        """Predict mask from sparse prompt + image embedding.

        Returns
        -------
        dict with:
            "mask": (B, 1, H, W) — sigmoid mask logits
            "iou_pred": (B, 1) — predicted IoU
            "uncertainty": (B, 1, H, W) or None — per-pixel variance (if enabled)
        """
        B = image_embedding.shape[0]
        # SAM mask decoder expects sparse prompt shape (B, N, 256)
        if sparse_prompt.dim() == 2:
            sparse_prompt = sparse_prompt.unsqueeze(1)  # (B, 1, 256)

        # Dense prompt: zeros (no dense prompt from user)
        dense_prompt = torch.zeros(B, 256, 64, 64,
                                   device=image_embedding.device,
                                   dtype=image_embedding.dtype)

        # Get positional encoding from SAM's mask decoder
        with torch.no_grad():
            image_pe = self.mask_decoder.no_mask_weight(
                torch.zeros(B, 256, 64, 64,
                           device=image_embedding.device,
                           dtype=image_embedding.dtype)
            )

        masks, iou_pred = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt,
            dense_prompt_embeddings=dense_prompt,
            multimask_output=False,  # single mask per query
        )

        result = {"mask": masks, "iou_pred": iou_pred}

        if self.predict_uncertainty:
            result["uncertainty"] = self.uncertainty_head(masks)

        return result


# ============================================================================
# Stub fallbacks (used when segment_anything is not installed)
# ============================================================================

class DenseEncoderStub(nn.Module):
    """Fallback when segment_anything is not available."""

    def __init__(self, name: str = "stub", freeze: bool = True,
                 no_grad_forward: bool = True, weights: str | None = None):
        super().__init__()
        self.name = name
        self.freeze = freeze
        self.no_grad_forward = no_grad_forward
        self.weights = weights
        self.register_buffer("_stub", torch.zeros(1), persistent=False)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return image


class MaskDecoderSkeleton(nn.Module):
    """Fallback MLP decoder when segment_anything is not available."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 predict_uncertainty: bool = True):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.predict_uncertainty = predict_uncertainty
        self.uncertainty_head = (
            nn.Linear(out_dim, out_dim) if predict_uncertainty else None
        )

    def forward(self, token_embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        mask_embedding = self.proj(token_embedding)
        output = {"mask_embedding": mask_embedding}
        if self.uncertainty_head is not None:
            output["uncertainty_embedding"] = self.uncertainty_head(mask_embedding)
        return output



class BoxRegressionHead(nn.Module):
    """Shared structure for `<BOX_S/M/L>` heads."""

    def __init__(self, in_dim: int, hidden_dim: int, n_class: int = 0):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
        )
        self.bbox_head = nn.Linear(hidden_dim, 4)
        self.class_head = nn.Linear(hidden_dim, n_class) if n_class > 0 else None

    def forward(self, token_embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.trunk(token_embedding)
        out = {"bbox": self.bbox_head(h)}
        if self.class_head is not None:
            out["class_logits"] = self.class_head(h)
        return out


class LocalizationHead(nn.Module):
    """Dual-output `<LOC>` head: pixel box + anatomy logits."""

    def __init__(self, in_dim: int, hidden_dim: int, n_anatomy: int = 0):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
        )
        self.bbox_head = nn.Linear(hidden_dim, 4)
        self.anatomy_head = nn.Linear(hidden_dim, n_anatomy) if n_anatomy > 0 else None

    def forward(self, token_embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.trunk(token_embedding)
        out = {"bbox": self.bbox_head(h)}
        if self.anatomy_head is not None:
            out["anatomy_logits"] = self.anatomy_head(h)
        return out


@registry.register_model("omnirad")
class OmniRad(MiniGPTv2):
    """Step-4 OmniRad model skeleton.

    This class intentionally focuses on the minimal architecture glue needed to:
    1. register a new `omnirad` arch;
    2. parse the OmniRad config sections;
    3. extend the tokenizer/model vocabulary with new routing tokens;
    4. attach placeholder dense / segmentation / localization heads;
    5. keep the existing MiniGPT-Med text-training path working.

    Full structured supervision and SAM-backed dense decoding are deferred to the
    next implementation steps.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/omnirad.yaml",
        "pretrain_llama3": "configs/models/omnirad_llama3.yaml",
    }

    DEFAULT_LOSS_WEIGHTS = {
        "text": 1.0,
        "det": 1.0,
        "loc": 1.0,
        "seg_bce": 2.0,
        "seg_dice": 0.5,
        "cardinality": 0.5,
        "cons_mb": 0.3,
        "cons_rd": 0.2,
        "cons_gs": 0.2,
        "align_anat": 0.5,
        "scale_w_s": 3.0,
        "scale_w_m": 1.5,
        "scale_w_l": 1.0,
        "uncertainty": 0.1,
    }

    def __init__(
        self,
        vit_model: str = "eva_clip_g",
        img_size: int = 448,
        drop_path_rate: float = 0,
        use_grad_checkpoint: bool = False,
        vit_precision: str = "fp16",
        freeze_vit: bool = True,
        llama_model: str = "",
        prompt_template: str = "[INST] {} [/INST]",
        max_txt_len: int = 300,
        end_sym: str = "\n",
        lora_r: int = 64,
        lora_target_modules: list[str] | None = None,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        chat_template: bool = False,
        use_grad_checkpoint_llm: bool = False,
        max_context_len: int = 3800,
        low_resource: bool = False,
        device_8bit: int = 0,
        expand_vocab: list[str] | None = None,
        dense_encoder: dict | None = None,
        mask_decoder: dict | None = None,
        loc_heads: dict | None = None,
        loss_weights: dict | None = None,
    ):
        lora_target_modules = lora_target_modules or ["q_proj", "v_proj"]
        super().__init__(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            llama_model=llama_model,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            lora_r=lora_r,
            lora_target_modules=lora_target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            chat_template=chat_template,
            use_grad_checkpoint_llm=use_grad_checkpoint_llm,
            max_context_len=max_context_len,
            low_resource=low_resource,
            device_8bit=device_8bit,
        )

        self.loss_weights = {**self.DEFAULT_LOSS_WEIGHTS, **(loss_weights or {})}
        self.special_tokens = tuple(expand_vocab or [])
        self.special_token_ids: dict[str, int] = {}
        self._expand_output_vocabulary(self.special_tokens)

        dense_encoder = dense_encoder or {}
        mask_decoder = mask_decoder or {}
        loc_heads = loc_heads or {}

        hidden_size = self.llama_model.config.hidden_size
        seg_hidden_dim = mask_decoder.get("hidden_dim", 512)
        seg_out_dim = mask_decoder.get("out_dim", 256)

        self.dense_encoder_cfg = dense_encoder
        self.mask_decoder_cfg = mask_decoder
        self.loc_heads_cfg = loc_heads

        # ---- Dense encoder: use SAM if available, else stub ----
        dense_name = dense_encoder.get("name", "sam_vit_h")
        dense_weights = dense_encoder.get("weights")
        dense_freeze = dense_encoder.get("freeze", True)
        dense_no_grad = dense_encoder.get("no_grad_forward", True)

        self._use_real_sam = _SAM_AVAILABLE and dense_name != "stub" and dense_weights

        if self._use_real_sam:
            self.dense_encoder = SamDenseEncoder(
                name=dense_name, weights=dense_weights,
                freeze=dense_freeze, no_grad_forward=dense_no_grad,
            )
        else:
            if not _SAM_AVAILABLE:
                logging.warning(
                    "segment_anything not installed; falling back to DenseEncoderStub. "
                    "Install with: pip install segment-anything"
                )
            self.dense_encoder = DenseEncoderStub(
                name=dense_name, freeze=dense_freeze,
                no_grad_forward=dense_no_grad, weights=dense_weights,
            )
        # SEG token projector: LLM hidden_size → SAM embedding dim (256)
        self.seg_projector = nn.Sequential(
            nn.Linear(hidden_size, seg_hidden_dim),
            nn.GELU(),
            nn.Linear(seg_hidden_dim, seg_out_dim),  # seg_out_dim = 256 for SAM
        )

        # ---- Mask decoder: use SAM if available, else skeleton ----
        if self._use_real_sam:
            self.mask_decoder = SamMaskDecoder(
                name=dense_name,
                weights=dense_weights,
                predict_uncertainty=mask_decoder.get("predict_uncertainty", True),
            )
        else:
            self.mask_decoder = MaskDecoderSkeleton(
                in_dim=seg_out_dim,
                hidden_dim=seg_hidden_dim,
                out_dim=seg_out_dim,
                predict_uncertainty=mask_decoder.get("predict_uncertainty", True),
            )
        self.box_heads = nn.ModuleDict({
            "box_s": self._build_box_head(loc_heads.get("box_s", {}), hidden_size),
            "box_m": self._build_box_head(loc_heads.get("box_m", {}), hidden_size),
            "box_l": self._build_box_head(loc_heads.get("box_l", {}), hidden_size),
        })
        self.loc_head = self._build_loc_head(loc_heads.get("loc", {}), hidden_size)

        logging.info(
            "Initialized OmniRad skeleton with %d special tokens: %s",
            len(self.special_tokens),
            list(self.special_tokens),
        )

    def _build_box_head(self, cfg: dict, hidden_size: int) -> BoxRegressionHead:
        in_dim = cfg.get("in_dim", hidden_size)
        hidden_dim = cfg.get("hidden_dim", 512)
        n_class = cfg.get("n_class", 0)
        return BoxRegressionHead(in_dim=in_dim, hidden_dim=hidden_dim, n_class=n_class)

    def _build_loc_head(self, cfg: dict, hidden_size: int) -> LocalizationHead:
        in_dim = cfg.get("in_dim", hidden_size)
        hidden_dim = cfg.get("hidden_dim", 512)
        n_anatomy = cfg.get("n_anatomy", 0)
        return LocalizationHead(in_dim=in_dim, hidden_dim=hidden_dim, n_anatomy=n_anatomy)

    @staticmethod
    def _unwrap_lm_head_cast(model):
        """Unwrap CastOutputToFloat from lm_head at the correct nesting level.

        The LLM may be wrapped by LoRA (PeftModel), so lm_head could be at
        model.base_model.model.lm_head rather than model.lm_head directly.
        We drill down to the innermost model and unwrap there.
        """
        target = model
        # Drill through PeftModel.base_model.model nesting
        while hasattr(target, "base_model") and hasattr(target.base_model, "model"):
            target = target.base_model.model

        if hasattr(target, "lm_head") and \
           type(target.lm_head).__name__ == "CastOutputToFloat":
            _inner = next(target.lm_head.children())
            target.lm_head = _inner
            logging.info("Unwrapped CastOutputToFloat from lm_head for resize_token_embeddings")

    def _expand_output_vocabulary(self, special_tokens: tuple[str, ...]):
        if not special_tokens:
            return

        added = self.llama_tokenizer.add_special_tokens(
            {"additional_special_tokens": list(special_tokens)}
        )
        if added > 0:
            # ★ Unwrap CastOutputToFloat wrapper on lm_head (it lacks .weight,
            # breaking resize_token_embeddings).  The wrapper only casts output
            # from fp16→fp32, which AMP handles automatically during training.
            self._unwrap_lm_head_cast(self.llama_model)

            self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))
            self._set_output_embeddings_trainable()

            self._set_output_embeddings_trainable()

        self.special_token_ids = {
            token: self.llama_tokenizer.convert_tokens_to_ids(token)
            for token in special_tokens
        }

    def _set_output_embeddings_trainable(self):
        input_embeddings = self.llama_model.get_input_embeddings()
        if input_embeddings is not None and hasattr(input_embeddings, "weight"):
            input_embeddings.weight.requires_grad = True

        output_embeddings = self.llama_model.get_output_embeddings()
        if output_embeddings is not None and hasattr(output_embeddings, "weight"):
            output_embeddings.weight.requires_grad = True

    def encode_dense_image(self, image: torch.Tensor) -> torch.Tensor:
        """Hook point for the future SAM-backed dense branch."""
        if self.dense_encoder_cfg.get("no_grad_forward", True):
            with torch.no_grad():
                return self.dense_encoder(image)
        return self.dense_encoder(image)

    @torch.no_grad()
    def generate(
        self,
        images,
        texts,
        num_beams=1,
        max_new_tokens=20,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1,
        length_penalty=1,
        temperature=1,
        do_sample=False,
        stop_words_ids=None,
        return_masks: bool = False,
    ):
        """OmniRad generate — preserves special tokens in output.

        Returns a list of dicts with keys:
            "text": str — decoded text with special tokens preserved
            "seg_count": int — number of <SEG> tokens
            "box_tokens": list[str] — ["<BOX_S>", "<BOX_M>", ...]
            "has_loc": bool — whether <LOC> was emitted
            "masks": list[torch.Tensor]  — only when ``return_masks=True``
                Each tensor has shape ``(H, W)`` in [0, 1] (sigmoid probability).
                If a sample emits K ``<SEG>`` tokens, K masks are returned in
                emission order.  Empty list when K = 0.
        """
        if stop_words_ids is None:
            # Auto-detect the eos token id for the current tokenizer.
            # LLaMA-2: </s> = 2; LLaMA-3: <|eot_id|> = 128009 or <|end_of_text|> = 128001.
            eos_id = self.llama_tokenizer.eos_token_id
            stop_words_ids = [eos_id if eos_id is not None else 2]

        from minigpt4.conversation.conversation import StoppingCriteriaSub
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(
            stops=[torch.tensor([i]).to(self.device) for i in stop_words_ids])])

        img_embeds, atts_img = self.encode_img(images.to(self.device))
        image_lists = [[image_emb[None]] for image_emb in img_embeds]
        batch_embs = [self.get_context_emb(text, img_list) for text, img_list in zip(texts, image_lists)]

        batch_size = len(batch_embs)
        max_len = max([emb.shape[1] for emb in batch_embs])
        emb_dim = batch_embs[0].shape[2]
        dtype = batch_embs[0].dtype
        device = batch_embs[0].device

        embs = torch.zeros([batch_size, max_len, emb_dim], dtype=dtype, device=device)
        attn_mask = torch.zeros([batch_size, max_len], dtype=torch.int, device=device)
        for i, emb in enumerate(batch_embs):
            emb_len = emb.shape[1]
            embs[i, -emb_len:] = emb[0]
            attn_mask[i, -emb_len:] = 1

        with self.maybe_autocast():
            outputs = self.llama_model.generate(
                inputs_embeds=embs,
                attention_mask=attn_mask,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                length_penalty=length_penalty,
                temperature=temperature,
                do_sample=do_sample,
                min_length=min_length,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )

        # ------------------------------------------------------------------
        # Optional second pass: recover <SEG> hidden states & decode masks.
        # We append the *generated* token embeddings to the original prompt
        # embeddings and run the LLM once more with output_hidden_states=True.
        # The last-layer hidden state at every <SEG> position is then fed
        # through ``seg_projector`` + ``mask_decoder`` (or stub) to produce
        # a binary mask per emission.
        # ------------------------------------------------------------------
        per_sample_masks: list[list[torch.Tensor]] = [[] for _ in range(batch_size)]
        if return_masks:
            per_sample_masks = self._decode_masks_from_outputs(
                outputs=outputs,
                prompt_embs=embs,
                prompt_attn_mask=attn_mask,
                images=images,
            )

        results = []
        for b_idx, output_token in enumerate(outputs):
            if output_token[0] == 0:
                output_token = output_token[1:]

            # P1-11: use skip_special_tokens=False to preserve <SEG>/<BOX_S/M/L>/<LOC>
            raw_text = self.llama_tokenizer.decode(output_token, skip_special_tokens=False)
            # Strip end-of-sequence markers for both LLaMA-2 and LLaMA-3.
            for _eos in ["</s>", "<|eot_id|>", "<|end_of_text|>"]:
                raw_text = raw_text.split(_eos)[0]
            # Strip BOS variants.
            for _bos in ["<s>", "<|begin_of_text|>"]:
                raw_text = raw_text.replace(_bos, "")
            # Extract assistant response.
            if "[/INST]" in raw_text:
                raw_text = raw_text.split(r'[/INST]')[-1].strip()
            elif "<|end_header_id|>" in raw_text:
                # LLaMA-3 header format: ...assistant<|end_header_id|>\n\n{response}
                parts = raw_text.split("<|end_header_id|>")
                raw_text = parts[-1].strip() if parts else raw_text

            # Also produce a clean text version (without special tokens) for backward compat
            clean_text = self.llama_tokenizer.decode(output_token, skip_special_tokens=True)
            for _eos in ["</s>", "<|eot_id|>", "<|end_of_text|>"]:
                clean_text = clean_text.split(_eos)[0]
            for _bos in ["<s>", "<|begin_of_text|>"]:
                clean_text = clean_text.replace(_bos, "")
            if "[/INST]" in clean_text:
                clean_text = clean_text.split(r'[/INST]')[-1].strip()
            elif "<|end_header_id|>" in clean_text:
                parts = clean_text.split("<|end_header_id|>")
                clean_text = parts[-1].strip() if parts else clean_text

            # Parse special token counts
            seg_count = raw_text.count("<SEG>")
            box_tokens = []
            for token in ["<BOX_S>", "<BOX_M>", "<BOX_L>"]:
                box_tokens.extend([token] * raw_text.count(token))
            has_loc = "<LOC>" in raw_text

            entry = {
                "text": clean_text,
                "raw_text": raw_text,
                "seg_count": seg_count,
                "box_tokens": box_tokens,
                "has_loc": has_loc,
            }
            if return_masks:
                entry["masks"] = per_sample_masks[b_idx]
            results.append(entry)

        return results

    @torch.no_grad()
    def _decode_masks_from_outputs(
        self,
        outputs: torch.Tensor,
        prompt_embs: torch.Tensor,
        prompt_attn_mask: torch.Tensor,
        images: torch.Tensor,
    ) -> list[list[torch.Tensor]]:
        """Second-pass mask decoding from a finished ``generate`` call.

        For every sample in the batch, locate every ``<SEG>`` token in the
        generated sequence and run the mask decoder once per occurrence.

        Returns
        -------
        list[list[Tensor]]
            ``out[b]`` is a list of ``(H, W)`` sigmoid-probability tensors,
            one per ``<SEG>`` emission for sample ``b``.  Empty list if the
            sample emitted no ``<SEG>``.
        """
        seg_id = self.special_token_ids.get("<SEG>", -1)
        batch_size = outputs.shape[0]
        per_sample_masks: list[list[torch.Tensor]] = [[] for _ in range(batch_size)]

        if seg_id < 0:
            return per_sample_masks
        # Quick exit: no SEG anywhere.
        if not (outputs == seg_id).any():
            return per_sample_masks

        # ---- Re-embed the generated tokens & concat to the prompt ----
        # Strip leading 0 / pad if present, but keep batch alignment via right-pad.
        gen_ids = outputs.clone()
        # Replace any pad (-1 or pad_token_id) with eos for safe embedding lookup.
        pad_id = getattr(self.llama_tokenizer, "pad_token_id", None) or 0
        gen_ids = gen_ids.masked_fill(gen_ids < 0, pad_id)
        gen_embeds = self.llama_model.get_input_embeddings()(gen_ids)
        gen_attn = (outputs != pad_id).to(prompt_attn_mask.dtype)

        full_embeds = torch.cat([prompt_embs, gen_embeds], dim=1)
        full_attn = torch.cat([prompt_attn_mask, gen_attn], dim=1)

        with self.maybe_autocast():
            llm_out = self.llama_model(
                inputs_embeds=full_embeds,
                attention_mask=full_attn,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden_states = llm_out.hidden_states[-1]  # (B, S_full, hidden)

        # SEG positions are in the *generated* segment; map them to full-seq offset.
        prompt_len = prompt_embs.shape[1]
        seg_positions_per_sample = []
        for b in range(batch_size):
            # boolean mask over gen tokens
            seg_mask = (outputs[b] == seg_id)
            positions = seg_mask.nonzero(as_tuple=True)[0] + prompt_len
            seg_positions_per_sample.append(positions)

        # Optionally compute dense image embedding once (SAM path)
        image_embedding = None
        if self._use_real_sam:
            image_embedding = self.encode_dense_image(images.to(self.device))
            # encode_dense_image expects 1024-resolution input; SamDenseEncoder
            # internally resizes — so this matches forward-pass behavior.

        # ---- Decode mask per <SEG> emission ----
        for b, positions in enumerate(seg_positions_per_sample):
            if positions.numel() == 0:
                continue
            for pos in positions:
                seg_hidden = hidden_states[b, pos, :]            # (hidden,)
                sparse = self.seg_projector(seg_hidden.unsqueeze(0))  # (1, 256)
                if self._use_real_sam and image_embedding is not None:
                    img_emb_b = image_embedding[b:b + 1]        # (1, 256, 64, 64)
                    decoder_out = self.mask_decoder(sparse, img_emb_b)
                    mask_logits = decoder_out["mask"]            # (1,1,H,W)
                    mask_prob = torch.sigmoid(mask_logits.float())
                    per_sample_masks[b].append(
                        mask_prob.squeeze(0).squeeze(0).detach().cpu()
                    )
                else:
                    # Stub fallback: no real geometry; emit a tiny placeholder.
                    per_sample_masks[b].append(
                        torch.zeros(64, 64, dtype=torch.float32)
                    )

        return per_sample_masks



    def extract_structured_targets(self, samples: dict) -> dict[str, Any]:
        """Normalize optional structured supervision fields from the dataloader.

        Existing MiniGPT-Med datasets only provide text targets. Group-US samples
        additionally provide variable-length dense targets, which are kept as
        per-sample lists because boxes and masks have different cardinalities.
        """
        has_structured = samples.get("has_structured_supervision")
        boxes = samples.get("boxes")
        mask_paths = samples.get("mask_paths")
        k_targets = samples.get("K")
        anatomy_regions = samples.get("anatomy_regions")
        box_scales = samples.get("box_scales")

        if has_structured is None:
            batch_size = len(samples.get("answer", []))
            device = self.device
            has_structured = torch.zeros(batch_size, dtype=torch.bool, device=device)

        return {
            "has_structured_supervision": has_structured,
            "boxes": boxes,
            "box_scales": box_scales,
            "anatomy_regions": anatomy_regions,
            "mask_paths": mask_paths,
            "K": k_targets,
        }

    def _extract_special_token_hidden_states(
        self,
        hidden_states: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict[str, list[torch.Tensor]]:
        """Extract last-layer hidden states at special-token positions.

        Parameters
        ----------
        hidden_states : ``(batch, seq_len, hidden_dim)`` — last-layer LLM hidden states
        targets : ``(batch, seq_len)`` — token IDs with -100 for non-target positions

        Returns
        -------
        dict mapping token-name → list of ``(hidden_dim,)`` tensors, one per occurrence,
        batched across the batch dimension.  E.g. ``{"<SEG>": [h1, h2, ...]}``.
        """
        result: dict[str, list[torch.Tensor]] = {}
        for token_name, token_id in self.special_token_ids.items():
            if token_id < 0:
                continue
            mask = (targets == token_id)  # (batch, seq_len)
            collected = []
            for b in range(mask.shape[0]):
                positions = mask[b].nonzero(as_tuple=True)[0]
                for pos in positions:
                    collected.append(hidden_states[b, pos, :])
            result[token_name] = collected
        return result

    def compute_auxiliary_losses(
        self,
        samples: dict,
        text_loss: torch.Tensor,
        structured_targets: dict[str, Any],
        special_hidden_states: dict[str, list[torch.Tensor]] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute detection / localization / segmentation / consistency losses.

        When no structured supervision is present (legacy datasets) or no special
        tokens were emitted in the current batch, all losses default to zero.
        """
        zero = text_loss.new_zeros(())
        has_structured = structured_targets.get("has_structured_supervision")

        # If no structured data in this batch, return zeros immediately
        if not (torch.is_tensor(has_structured) and has_structured.any()):
            return {"det": zero, "loc": zero, "seg": zero, "cons": zero}

        if special_hidden_states is None:
            return {"det": zero, "loc": zero, "seg": zero, "cons": zero}

        # ---- Detection loss (BOX_S/M/L) ----
        det_loss = self._compute_det_loss(special_hidden_states, structured_targets, text_loss)

        # ---- Localization loss (LOC) ----
        loc_loss = self._compute_loc_loss(special_hidden_states, structured_targets, text_loss)

        # ---- Segmentation loss (SEG) ----
        seg_loss = self._compute_seg_loss(special_hidden_states, structured_targets, samples, text_loss)

        # ---- Consistency loss (mask-box geometric) ----
        cons_loss = self._compute_cons_loss(special_hidden_states, structured_targets, samples, text_loss)

        return {"det": det_loss, "loc": loc_loss, "seg": seg_loss, "cons": cons_loss}

    def _compute_cons_loss(
        self,
        special_hidden_states: dict[str, list[torch.Tensor]],
        structured_targets: dict[str, Any],
        samples: dict,
        ref_loss: torch.Tensor,
    ) -> torch.Tensor:
        """Mask-Box geometric consistency: tight bbox from predicted mask should
        match the predicted bbox from <BOX> tokens.

        L_cons = ||phi(M_pred) - b_pred||_1

        where phi extracts the tight enclosing box from a binary mask.
        Only computed when both <SEG> and <BOX> tokens are present in the same batch.
        """
        seg_preds = special_hidden_states.get("<SEG>", [])
        box_preds = []
        for token_name in ["<BOX_S>", "<BOX_M>", "<BOX_L>"]:
            box_preds.extend(special_hidden_states.get(token_name, []))

        if not seg_preds or not box_preds:
            return ref_loss.new_zeros(())

        # Only compute if SAM is available (need real mask to extract tight box)
        if not self._use_real_sam:
            return ref_loss.new_zeros(())

        images = samples.get("image")
        if images is None:
            return ref_loss.new_zeros(())

        image_embedding = self.encode_dense_image(images)
        if image_embedding is None:
            return ref_loss.new_zeros(())

        total_loss = ref_loss.new_zeros(())
        n_pairs = 0

        n_pairs_to_check = min(len(seg_preds), len(box_preds))
        for idx in range(n_pairs_to_check):
            seg_hidden = seg_preds[idx]
            box_hidden = box_preds[idx]

            # Get mask prediction
            sparse_prompt = self.seg_projector(seg_hidden.unsqueeze(0))
            img_emb = image_embedding[0:1]
            decoder_out = self.mask_decoder(sparse_prompt, img_emb)
            pred_mask = decoder_out["mask"].squeeze()  # (H, W)
            pred_prob = torch.sigmoid(pred_mask)

            # Extract tight bbox from mask (soft version for differentiability)
            H, W = pred_prob.shape
            grid_y = torch.arange(H, device=pred_prob.device, dtype=pred_prob.dtype) / H
            grid_x = torch.arange(W, device=pred_prob.device, dtype=pred_prob.dtype) / W

            # Weighted mean → center
            weight = pred_prob + 1e-8
            cx = (grid_x.unsqueeze(0) * weight.sum(dim=0)).sum() / weight.sum()
            cy = (grid_y.unsqueeze(1) * weight.sum(dim=1)).sum() / weight.sum()

            # Weighted extent (approximate tight box)
            x_min = (pred_prob.sum(dim=0) > 0.1).float().argmax().float() / W
            x_max = 1.0 - (pred_prob.sum(dim=0).flip(0) > 0.1).float().argmax().float() / W
            y_min = (pred_prob.sum(dim=1) > 0.1).float().argmax().float() / H
            y_max = 1.0 - (pred_prob.sum(dim=1).flip(0) > 0.1).float().argmax().float() / H

            mask_bbox = torch.stack([x_min, y_min, x_max, y_max])

            # Get box prediction
            scale_key = {"<BOX_S>": "box_s", "<BOX_M>": "box_m", "<BOX_L>": "box_l"}
            # Find which scale this box prediction came from
            head = self.box_heads["box_m"]  # default
            box_out = head(box_hidden.unsqueeze(0))
            pred_bbox = torch.sigmoid(box_out["bbox"].squeeze(0))  # (4,) in [0,1]

            l1 = torch.nn.functional.l1_loss(mask_bbox, pred_bbox)
            total_loss = total_loss + l1
            n_pairs += 1

        if n_pairs == 0:
            return ref_loss.new_zeros(())
        return total_loss / n_pairs

    def _compute_det_loss(
        self,
        special_hidden_states: dict[str, list[torch.Tensor]],
        structured_targets: dict[str, Any],
        ref_loss: torch.Tensor,
    ) -> torch.Tensor:
        """Scale-aware detection loss for <BOX_S/M/L> tokens."""
        scale_tokens = {"<BOX_S>": "box_s", "<BOX_M>": "box_m", "<BOX_L>": "box_l"}
        scale_weights = {"box_s": self.loss_weights.get("scale_w_s", 3.0),
                         "box_m": self.loss_weights.get("scale_w_m", 1.5),
                         "box_l": self.loss_weights.get("scale_w_l", 1.0)}

        boxes_gt = structured_targets.get("boxes")  # list[Tensor] per sample
        box_scales_gt = structured_targets.get("box_scales")  # list[list[str]]
        if boxes_gt is None:
            return ref_loss.new_zeros(())

        total_loss = ref_loss.new_zeros(())
        n_valid = 0

        # Flatten GT boxes across batch with scale labels
        gt_boxes_flat = []
        gt_scales_flat = []
        for b, (boxes_b, scales_b) in enumerate(zip(boxes_gt, box_scales_gt or [])):
            if not torch.is_tensor(boxes_b) or boxes_b.shape[0] == 0:
                continue
            for idx in range(boxes_b.shape[0]):
                gt_boxes_flat.append(boxes_b[idx])
                gt_scales_flat.append(scales_b[idx] if idx < len(scales_b) else "M")

        # Match predictions to GT by order
        pred_idx = 0
        for token_name, head_key in scale_tokens.items():
            preds = special_hidden_states.get(token_name, [])
            head = self.box_heads[head_key]
            for pred_hidden in preds:
                if pred_idx >= len(gt_boxes_flat):
                    break
                out = head(pred_hidden.unsqueeze(0))
                pred_bbox = out["bbox"].squeeze(0)  # (4,)
                gt_bbox = gt_boxes_flat[pred_idx].to(pred_bbox.device).float()
                # Normalize GT bbox to [0,1] if it looks like pixel coords
                # (dataset returns pixel coords; we normalize by image size 448)
                gt_bbox = gt_bbox / 448.0
                gt_bbox = gt_bbox.clamp(0.0, 1.0)

                l1 = torch.nn.functional.l1_loss(pred_bbox, gt_bbox)
                # Simple GIoU approximation (1D since we don't have full 2D GIoU here)
                giou = 1.0 - l1  # simplified
                scale_w = scale_weights[head_key]
                total_loss = total_loss + scale_w * (
                    self.loss_weights.get("l1_det", 1.0) * l1
                )
                pred_idx += 1
                n_valid += 1

        if n_valid == 0:
            return ref_loss.new_zeros(())
        return total_loss / n_valid

    def _compute_loc_loss(
        self,
        special_hidden_states: dict[str, list[torch.Tensor]],
        structured_targets: dict[str, Any],
        ref_loss: torch.Tensor,
    ) -> torch.Tensor:
        """Localization loss for <LOC> token (pixel bbox + anatomy)."""
        loc_preds = special_hidden_states.get("<LOC>", [])
        if not loc_preds:
            return ref_loss.new_zeros(())

        boxes_gt = structured_targets.get("boxes")
        anatomy_regions_gt = structured_targets.get("anatomy_regions")
        if boxes_gt is None:
            return ref_loss.new_zeros(())

        total_loss = ref_loss.new_zeros(())
        n_valid = 0

        # Flatten GT
        gt_boxes_flat = []
        gt_anatomy_flat = []
        for b, boxes_b in enumerate(boxes_gt):
            if not torch.is_tensor(boxes_b) or boxes_b.shape[0] == 0:
                continue
            anatomy_b = (anatomy_regions_gt[b] if anatomy_regions_gt and b < len(anatomy_regions_gt) else [])
            for idx in range(boxes_b.shape[0]):
                gt_boxes_flat.append(boxes_b[idx])
                gt_anatomy_flat.append(anatomy_b[idx] if idx < len(anatomy_b) else None)

        for pred_idx, pred_hidden in enumerate(loc_preds):
            if pred_idx >= len(gt_boxes_flat):
                break
            out = self.loc_head(pred_hidden.unsqueeze(0))
            pred_bbox = out["bbox"].squeeze(0)
            gt_bbox = gt_boxes_flat[pred_idx].to(pred_bbox.device).float() / 448.0
            gt_bbox = gt_bbox.clamp(0.0, 1.0)

            l1 = torch.nn.functional.l1_loss(pred_bbox, gt_bbox)
            total_loss = total_loss + l1

            # Anatomy CE if head supports it and GT region is available
            if "anatomy_logits" in out and gt_anatomy_flat[pred_idx] is not None:
                # Simple: treat anatomy region string as a class index via hash
                # (proper anatomy label mapping will be added later)
                pass

            n_valid += 1

        if n_valid == 0:
            return ref_loss.new_zeros(())
        return total_loss / n_valid

    def _compute_seg_loss(
        self,
        special_hidden_states: dict[str, list[torch.Tensor]],
        structured_targets: dict[str, Any],
        samples: dict,
        ref_loss: torch.Tensor,
    ) -> torch.Tensor:
        """Segmentation loss for <SEG> tokens (BCE + Dice + cardinality).

        When SAM is available, the flow is:
            SEG hidden → seg_projector (→256d) → SamMaskDecoder → mask logits
            → BCE + Dice against GT mask

        When SAM is not available, only cardinality penalty is computed.
        """
        seg_preds = special_hidden_states.get("<SEG>", [])
        masks_gt = samples.get("masks")  # list[list[Tensor]] from UnifiedUSDataset
        k_gt = structured_targets.get("K")  # Tensor (batch,)

        total_loss = ref_loss.new_zeros(())
        n_valid = 0

        if seg_preds:
            # Flatten GT masks across batch
            gt_masks_flat = []
            if masks_gt:
                for masks_b in masks_gt:
                    if masks_b:
                        gt_masks_flat.extend(masks_b)

            # Get dense image embedding if SAM is available
            image_embedding = None
            if self._use_real_sam:
                images = samples.get("image")
                if images is not None:
                    image_embedding = self.encode_dense_image(images)

            for pred_idx, pred_hidden in enumerate(seg_preds):
                if pred_idx >= len(gt_masks_flat):
                    break

                gt_mask = gt_masks_flat[pred_idx].to(pred_hidden.device).float()

                if self._use_real_sam and image_embedding is not None:
                    # ---- Real SAM path ----
                    # Project SEG hidden → 256-d sparse prompt
                    sparse_prompt = self.seg_projector(pred_hidden.unsqueeze(0))  # (1, 256)

                    # Determine which batch sample this mask belongs to
                    # (simplified: use first image embedding)
                    img_emb = image_embedding[0:1]  # (1, 256, 64, 64)

                    decoder_out = self.mask_decoder(sparse_prompt, img_emb)
                    pred_mask = decoder_out["mask"]  # (1, 1, H, W) or (1, 1, 256, 256)

                    # Resize prediction to match GT mask size
                    if pred_mask.shape[-2:] != gt_mask.shape[-2:]:
                        pred_mask = F.interpolate(
                            pred_mask, size=gt_mask.shape[-2:],
                            mode="bilinear", align_corners=False,
                        )
                    pred_mask = pred_mask.squeeze()  # (H, W)
                    gt_mask = gt_mask.squeeze()

                    # BCE loss
                    bce_loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask)
                    # Dice loss
                    pred_prob = torch.sigmoid(pred_mask)
                    intersection = (pred_prob * gt_mask).sum()
                    dice_loss = 1.0 - (2.0 * intersection + 1.0) / (
                        pred_prob.sum() + gt_mask.sum() + 1.0
                    )

                    seg_loss = (
                        self.loss_weights.get("seg_bce", 2.0) * bce_loss
                        + self.loss_weights.get("seg_dice", 0.5) * dice_loss
                    )

                    # Uncertainty / heteroscedastic loss (AMU-Seg Innovation II)
                    if "uncertainty" in decoder_out:
                        log_variance = decoder_out["uncertainty"].squeeze()
                        # Heteroscedastic BCE: down-weight uncertain pixels
                        # L = (1 / 2σ²) × BCE + (1/2) × log(σ²)
                        variance = torch.exp(log_variance.clamp(-10, 10))  # σ² = exp(log_var)
                        weighted_bce = (bce_loss * (1.0 / (2.0 * variance + 1e-6))).mean()
                        reg_term = 0.5 * log_variance.mean()
                        unc_loss = weighted_bce + reg_term
                        seg_loss = seg_loss + 0.1 * unc_loss  # small weight for uncertainty
                else:
                    # ---- Stub path: no real mask, use cardinality only ----
                    seg_loss = ref_loss.new_zeros(())

                total_loss = total_loss + seg_loss
                n_valid += 1

            if n_valid > 0:
                total_loss = total_loss / n_valid

        # Cardinality loss: penalize K mismatch
        if torch.is_tensor(k_gt):
            n_emitted = len(seg_preds)
            expected_k = k_gt.sum().item() if k_gt.numel() > 0 else 0
            if n_emitted != expected_k:
                card_penalty = abs(n_emitted - expected_k) / max(expected_k, 1)
                total_loss = total_loss + self.loss_weights.get("cardinality", 0.5) * card_penalty

        return total_loss

    def forward(self, samples: dict, reduction: str = "mean") -> dict[str, torch.Tensor]:
        """OmniRad forward with special-token hidden state extraction and aux losses.

        This method overrides MiniGPTBase.forward to:
        1. Prepare embeddings (inherited logic)
        2. Run LLM with output_hidden_states=True
        3. Extract hidden states at <SEG>/<BOX_S/M/L>/<LOC> positions
        4. Compute text loss + auxiliary det/loc/seg/cons losses
        """
        structured_targets = self.extract_structured_targets(samples)

        # ---- Step 1: Prepare embeddings (same as MiniGPTBase.forward) ----
        cond_embeds, cond_atts, regress_embeds, regress_atts, part_targets = \
            self.preparing_embedding(samples)

        inputs_embeds, attention_mask, input_lens = \
            self.concat_emb_input_output(cond_embeds, cond_atts, regress_embeds, regress_atts)

        # BOS token
        bos = torch.ones_like(part_targets[:, :1]) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        bos_atts = cond_atts[:, :1]

        inputs_embeds = torch.cat([bos_embeds, inputs_embeds], dim=1)
        attention_mask = torch.cat([bos_atts, attention_mask], dim=1)

        # Build targets (same as parent)
        targets = torch.ones([inputs_embeds.shape[0], inputs_embeds.shape[1]],
                             dtype=torch.long).to(self.device).fill_(-100)
        for i, target in enumerate(part_targets):
            targets[i, input_lens[i] + 1:input_lens[i] + len(target) + 1] = target

        # ---- Step 2: Run LLM with hidden states ----
        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
                reduction=reduction,
                output_hidden_states=True,
            )
        text_loss = outputs.loss

        # ---- Step 3: Extract special token hidden states ----
        special_hidden_states: dict[str, list[torch.Tensor]] = {}
        if self.special_token_ids and outputs.hidden_states is not None:
            last_hidden = outputs.hidden_states[-1]  # (batch, seq_len, hidden_dim)
            special_hidden_states = self._extract_special_token_hidden_states(
                last_hidden, targets
            )

        # ---- Step 4: Compute auxiliary losses ----
        aux_losses = self.compute_auxiliary_losses(
            samples, text_loss, structured_targets, special_hidden_states
        )

        total_loss = text_loss * self.loss_weights.get("text", 1.0)
        total_loss = total_loss + aux_losses["det"] * self.loss_weights.get("det", 1.0)
        total_loss = total_loss + aux_losses["loc"] * self.loss_weights.get("loc", 1.0)
        total_loss = total_loss + aux_losses["seg"]
        total_loss = total_loss + aux_losses["cons"] * self.loss_weights.get("cons_mb", 0.3)

        result = {
            "loss": total_loss,
            "loss_text": text_loss,
            "loss_det": aux_losses["det"],
            "loss_loc": aux_losses["loc"],
            "loss_seg": aux_losses["seg"],
            "loss_cons": aux_losses["cons"],
            "structured_targets": structured_targets,
            "special_token_ids": self.special_token_ids,
            "n_seg_tokens": len(special_hidden_states.get("<SEG>", [])),
            "n_box_tokens": sum(
                len(special_hidden_states.get(t, []))
                for t in ["<BOX_S>", "<BOX_M>", "<BOX_L>"]
            ),
            "n_loc_tokens": len(special_hidden_states.get("<LOC>", [])),
        }
        # Store for wandb logging
        self._last_forward_outputs = result
        return result




    @classmethod
    def from_config(cls, cfg):
        def _resolve(p):
            if not p or os.path.isabs(p):
                return os.path.normpath(p) if p else p
            try:
                from minigpt4.common.registry import registry
                abs_path = os.path.join(registry.get_path("repo_root"), p)
                return os.path.normpath(abs_path)       # ★ eliminate ".." / "." components
            except Exception:
                return p

        dense_encoder_cfg = _to_plain_dict(cfg.get("dense_encoder"))
        if dense_encoder_cfg and dense_encoder_cfg.get("weights"):
            dense_encoder_cfg["weights"] = _resolve(dense_encoder_cfg["weights"])

        model = cls(
            vit_model=cfg.get("vit_model", "eva_clip_g"),
            img_size=cfg.get("image_size", 448),
            drop_path_rate=cfg.get("drop_path_rate", 0),
            use_grad_checkpoint=cfg.get("use_grad_checkpoint", False),
            vit_precision=cfg.get("vit_precision", "fp16"),
            freeze_vit=cfg.get("freeze_vit", True),
            llama_model=_resolve(cfg.get("llama_model", "")),
            prompt_template=cfg.get("prompt_template", "[INST] {} [/INST]"),
            max_txt_len=cfg.get("max_txt_len", 300),
            end_sym=cfg.get("end_sym", "\n"),
            lora_r=cfg.get("lora_r", 64),
            lora_target_modules=cfg.get("lora_target_modules", ["q_proj", "v_proj"]),
            lora_alpha=cfg.get("lora_alpha", 16),
            lora_dropout=cfg.get("lora_dropout", 0.05),
            chat_template=cfg.get("chat_template", False),
            use_grad_checkpoint_llm=cfg.get("use_grad_checkpoint_llm", False),
            max_context_len=cfg.get("max_context_len", 3800),
            low_resource=cfg.get("low_resource", False),
            device_8bit=cfg.get("device_8bit", 0),
            expand_vocab=list(cfg.get("expand_vocab", [])),
            dense_encoder=dense_encoder_cfg,
            mask_decoder=_to_plain_dict(cfg.get("mask_decoder")),
            loc_heads=_to_plain_dict(cfg.get("loc_heads")),
            loss_weights=_to_plain_dict(cfg.get("loss_weights")),
        )

        ckpt_path = _resolve(cfg.get("ckpt", ""))
        if ckpt_path:
            logging.info("Load OmniRad checkpoint: %s", ckpt_path)
            ckpt = torch.load(ckpt_path, map_location="cpu")
            state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            # P2-19: strict=False because frozen modules (SAM ViT-H, EVA-ViT-G) are
            # not saved in training checkpoints (requires_grad=False filtered out).
            # Their weights must be loaded separately from the original pretrained files.
            msg = model.load_state_dict(state_dict, strict=False)
            if msg.missing_keys:
                logging.warning(
                    "OmniRad checkpoint missing keys (expected for frozen modules): %s",
                    msg.missing_keys[:20],  # show first 20 only
                )
            if msg.unexpected_keys:
                logging.warning("OmniRad checkpoint unexpected keys: %s", msg.unexpected_keys[:20])

        return model

