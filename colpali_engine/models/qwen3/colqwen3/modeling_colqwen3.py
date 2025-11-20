from typing import ClassVar, List, Optional

import torch
import torch.nn as nn
from transformers import Qwen3VLConfig, Qwen3VLModel


class ColQwen3(Qwen3VLModel):
    """
    ColQwen3 model implementation from the "ColPali: Efficient Document Retrieval with Vision Language Models" paper.
    """

    config_class = Qwen3VLConfig
    main_input_name = "doc_input_ids"
    _supports_flash_attn_2 = True

    def __init__(self, config: Qwen3VLConfig, mask_non_image_embeddings: bool = False):
        super().__init__(config=config)
        self.dim = 128
        # Qwen3VL has text_config that contains hidden_size
        text_config = config.get_text_config()
        self.custom_text_proj = nn.Linear(text_config.hidden_size, self.dim)
        self.padding_side = "left"
        self.mask_non_image_embeddings = mask_non_image_embeddings
        self.post_init()

    def forward(self, *args, **kwargs) -> torch.Tensor:
        # Handle the custom "pixel_values" input obtained with `ColQwen3Processor` through unpadding
        if "pixel_values" in kwargs and "image_grid_thw" in kwargs:
            # If pixel_values is None, remove it (Qwen3VL might not handle None explicitly if expected)
            if kwargs["pixel_values"] is None:
                del kwargs["pixel_values"]
            else:
                # Re-flatten pixel_values if they were padded by the processor
                # This logic mirrors ColQwen2's handling
                offsets = kwargs["image_grid_thw"][:, 1] * kwargs["image_grid_thw"][:, 2]  # (batch_size,)
                kwargs["pixel_values"] = torch.cat(
                    [pixel_sequence[:offset] for pixel_sequence, offset in zip(kwargs["pixel_values"], offsets)],
                    dim=0,
                )

        kwargs.pop("return_dict", True)
        kwargs.pop("output_hidden_states", None)
        kwargs.pop("use_cache", None)
        
        outputs = super().forward(*args, **kwargs, use_cache=False, output_hidden_states=True, return_dict=True)
        
        hidden_states = outputs.last_hidden_state
        proj = self.custom_text_proj(hidden_states)

        # L2 normalization
        proj = proj / proj.norm(dim=-1, keepdim=True)
        
        if "attention_mask" in kwargs:
             proj = proj * kwargs["attention_mask"].unsqueeze(-1)

        if "pixel_values" in kwargs and self.mask_non_image_embeddings:
            # Pools only the image embeddings
            # Assuming image_token_id is available in config
            if hasattr(self.config, "image_token_id"):
                image_mask = (kwargs["input_ids"] == self.config.image_token_id).unsqueeze(-1)
                proj = proj * image_mask
                
        return proj

    @property
    def patch_size(self) -> int:
        return self.visual.config.patch_size

    @property
    def spatial_merge_size(self) -> int:
        return self.visual.config.spatial_merge_size
