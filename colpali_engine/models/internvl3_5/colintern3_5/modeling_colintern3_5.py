# coding=utf-8
# ColIntern3.5: InternVL3.5-based late-interaction retriever (image/text encoders + 128-d projections)

from __future__ import annotations
from typing import ClassVar, Optional

import torch
from torch import nn
from transformers.models.internvl.modeling_internvl import (
    InternVLConfig,
    InternVLForConditionalGeneration,
    InternVLPreTrainedModel,
)


class ColIntern3_5(InternVLPreTrainedModel):  # noqa: N801
    """
    ColIntern3.5 wraps InternVL3.5 and exposes two embedding paths:

      • Text-only  (input_ids, attention_mask)  -> (B, L, 128)
      • Image-only (pixel_values)               -> (B, N_patches, 128)

    Both outputs are L2-normalized, ColBERT-style.
    """

    main_input_name: ClassVar[str] = "doc_input_ids"  # keeps HF Trainer happy for doc batches

    # Robust key remapping so HF InternVL checkpoints load into this wrapper (which nests InternVL twice: model.model.*)
    _checkpoint_conversion_mapping = {
        r"^module\.": "",
        r"^base_model\.model\.": "",

        # If checkpoint is InternVLForConditionalGeneration (usual): model.<submodule> -> model.model.<submodule>
        r"^model\.vision_tower": "model.model.vision_tower",
        r"^model\.language_model": "model.model.language_model",
        r"^model\.multi_modal_projector": "model.model.multi_modal_projector",
        r"^model\.lm_head": "model.lm_head",

        # If checkpoint is lower-level InternVLModel or other converters
        r"^vision_tower": "model.model.vision_tower",
        r"^vision_model": "model.model.vision_tower",
        r"^language_model": "model.model.language_model",
        r"^multi_modal_projector": "model.model.multi_modal_projector",

        # Idempotency (no-op if already model.model.*)
        r"^model\.model\.vision_tower": "model.model.vision_tower",
        r"^model\.model\.language_model": "model.model.language_model",
        r"^model\.model\.multi_modal_projector": "model.model.multi_modal_projector",
    }

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        kwargs.setdefault("key_mapping", cls._checkpoint_conversion_mapping)
        return super().from_pretrained(*args, **kwargs)

    def __init__(self, config: InternVLConfig, mask_non_image_embeddings: bool = False, output_dim: int = 128):
        super().__init__(config=config)

        # Keep the full HF backbone inside .model, like ColPali does
        self.model = InternVLForConditionalGeneration(config=config)

        # Propagate tied-weights bookkeeping for safe resize/tie
        lm = getattr(self.model, "language_model", None)
        if lm is not None and getattr(lm, "_tied_weights_keys", None):
            self._tied_weights_keys = [f"model.language_model.{k}" for k in lm._tied_weights_keys]

        # Retrieval head(s)
        self.dim = int(output_dim)
        txt_h = self.model.config.text_config.hidden_size
        vis_h = self.model.vision_tower.config.hidden_size
        self.custom_text_proj = nn.Linear(txt_h, self.dim, bias=False)
        self.custom_vision_proj = nn.Linear(vis_h, self.dim, bias=False)

        # Not doing generation
        self.model.lm_head = torch.nn.Identity()

        self.mask_non_image_embeddings = bool(mask_non_image_embeddings)
        self.post_init()

    # -----------------------
    # Convenience encoders
    # -----------------------
    @torch.no_grad()
    def embed_queries(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self._encode_text(input_ids, attention_mask)

    @torch.no_grad()
    def embed_documents(self, pixel_values: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self._encode_images(pixel_values)
    def floating_point_ops(self, inputs) -> int:
        """
        Very rough FLOPs estimator to satisfy HF Trainer logging.
        Counts text tokens or vision patches and scales by layer/hidden size.
        """
        # Count sequence "tokens"
        num_tokens = 0

        # Text tokens (sum of attention_mask)
        am = inputs.get("attention_mask", None)
        if am is not None:
            try:
                num_tokens += int(am.sum().item())
            except Exception:
                pass

        # Vision "tokens" (patches after merge)
        px = inputs.get("pixel_values", None)
        if px is not None and hasattr(px, "shape"):
            try:
                B, _, H, W = px.shape
                patch = self.patch_size
                merge = int(getattr(self.model, "spatial_merge_size", 2))
                n_x = (W // (patch * merge))
                n_y = (H // (patch * merge))
                num_tokens += int(B * n_x * n_y)
            except Exception:
                pass

        # If we couldn't compute anything, return 0 (Trainer will still proceed without warning)
        if num_tokens == 0:
            return 0

        # Pick a depth/width based on which path this batch looks like
        is_vision = ("pixel_values" in inputs) and ("input_ids" not in inputs)
        if is_vision:
            layers = int(self.model.vision_tower.config.num_hidden_layers)
            hidden = int(self.model.vision_tower.config.hidden_size)
        else:
            layers = int(self.model.config.text_config.num_hidden_layers)
            hidden = int(self.model.config.text_config.hidden_size)

        # Super coarse: per-layer per-token ~ O(hidden) scaling; constant factor not critical for logging
        return int(6 * num_tokens * hidden * layers)

    # -----------------------
    # Core encoding paths
    # -----------------------
    def _encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        dtype = self.dtype
        pixel_values = pixel_values.to(device=device, dtype=dtype)

        # Vision-only forward: go straight to the vision tower (no text tokens required)
        vision_out = self.model.vision_tower(pixel_values=pixel_values, return_dict=True)
        patch_states = vision_out.last_hidden_state[:, 1:, :]  # drop CLS -> (B, N_patches, H_vis)

        proj = self.custom_vision_proj(patch_states)           # (B, N_patches, 128)
        proj = proj / (proj.norm(dim=-1, keepdim=True) + 1e-12)
        return proj

    def _encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        input_ids = input_ids.to(device=device)
        attention_mask = attention_mask.to(device=device)

        # Text-only forward on the LLM
        outputs = self.model.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,                 # plays nicely with grad checkpointing
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = outputs.hidden_states[-1]                         # (B, L, H_txt)
        proj = self.custom_text_proj(hidden)                       # (B, L, 128)
        proj = proj / (proj.norm(dim=-1, keepdim=True) + 1e-12)
        proj = proj * attention_mask.to(dtype=proj.dtype).unsqueeze(-1)
        return proj

    def forward(
        self,
        *,
        # text
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        # images
        pixel_values: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Two mutually-exclusive modes:
          • Text-only  -> pass (input_ids, attention_mask) and no pixel_values
          • Image-only -> pass (pixel_values) and no input_ids

        Returns:
          torch.Tensor shaped (B, L_tokens_or_patches, 128)
        """
        has_text = input_ids is not None
        has_img = pixel_values is not None

        if has_text and has_img:
            raise ValueError("ColIntern3_5.forward accepts either text or images, not both simultaneously.")
        if not has_text and not has_img:
            raise ValueError("Provide either (input_ids, attention_mask) or pixel_values.")

        if has_img:
            return self._encode_images(pixel_values=pixel_values)

        if attention_mask is None:
            raise ValueError("attention_mask must be provided with input_ids.")
        return self._encode_text(input_ids=input_ids, attention_mask=attention_mask)

    # -------------
    # HF passthrough
    # -------------
    def get_input_embeddings(self):
        return self.model.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.model.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.model.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.model.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.language_model.get_decoder()

    def tie_weights(self):
        return self.model.language_model.tie_weights()

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None) -> nn.Embedding:
        model_embeds = self.model.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # keep config in sync
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.config.vocab_size = model_embeds.num_embeddings
        self.model.vocab_size = model_embeds.num_embeddings
        return model_embeds

    @property
    def patch_size(self) -> int:
        ps = self.model.vision_tower.config.patch_size
        return int(ps[0]) if isinstance(ps, (list, tuple)) else int(ps)
