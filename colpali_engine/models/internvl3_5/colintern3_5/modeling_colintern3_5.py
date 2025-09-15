from typing import ClassVar
import torch
from torch import nn
from transformers.models.internvl import (
    InternVLConfig,
    InternVLModel,
)

class ColIntern3_5(InternVLModel):  # noqa: N801
    """
    ColIntern3.5 model implementation for multi-vector retrieval, based on the InternVL3.5-1B vision-language backbone.
    Applies a linear projection to produce 128-dimensional token-wise embeddings for ColBERT late interaction.
    """
    main_input_name: ClassVar[str] = "input_ids"

    def __init__(self, config: InternVLConfig, mask_non_image_embeddings: bool = False):
        super().__init__(config)
        self.dim = 128
        text_hidden_size = config.text_config.hidden_size
        self.custom_text_proj = nn.Linear(text_hidden_size, self.dim)
        self.padding_side = "right"
        self.mask_non_image_embeddings = mask_non_image_embeddings
        self.post_init()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # Ensure dtype consistency for all params incl. projection
        torch_dtype = kwargs.get("torch_dtype", None)
        model = super().from_pretrained(*args, **kwargs)
        if torch_dtype is not None:
            model = model.to(dtype=torch_dtype)
            if hasattr(model, "custom_text_proj"):
                model.custom_text_proj = model.custom_text_proj.to(dtype=torch_dtype)
        return model

    def forward(self, *args, **kwargs) -> torch.Tensor:
        # Remove unsupported HF arguments if present
        kwargs.pop("return_dict", None)
        kwargs.pop("output_hidden_states", None)
        kwargs.pop("use_cache", None)

        # Forward through base InternVL model to obtain hidden states
        outputs = super().forward(*args, **kwargs, output_hidden_states=True, return_dict=True)
        last_hidden_states = outputs.last_hidden_state  # (batch_size, seq_length, hidden_size)

        # Project hidden states to low-dimensional embeddings
        proj = self.custom_text_proj(last_hidden_states)  # (batch_size, seq_length, 128)

        # L2 normalize the embeddings (avoid div by zero)
        proj = proj / proj.norm(dim=-1, keepdim=True).clamp(min=1e-12)

        # Mask out padding positions
        if "attention_mask" in kwargs and kwargs["attention_mask"] is not None:
            proj = proj * kwargs["attention_mask"].unsqueeze(-1).to(proj.dtype)

        # Optional masking of non-image tokens (OFF by default; patch-id heuristics are brittle)
        if "pixel_values" in kwargs and self.mask_non_image_embeddings:
            image_token_id = getattr(self.config, 'image_token_id', None)
            if image_token_id is not None and "input_ids" in kwargs:
                image_mask = (kwargs["input_ids"] == image_token_id).unsqueeze(-1)
                proj = proj * image_mask.to(proj.dtype)

        return proj

    @property
    def patch_size(self) -> int:
        ps = getattr(self.config.vision_config, "patch_size", 0)
        return ps[0] if isinstance(ps, (list, tuple)) else ps

    @property
    def spatial_merge_size(self) -> int:
        # Use the official downsample_ratio from config
        downsample_ratio = getattr(self.config, "downsample_ratio", 0.5)
        return int(1 / downsample_ratio)
