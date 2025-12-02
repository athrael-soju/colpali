from typing import ClassVar

import torch
from torch import nn

try:
    from transformers.models.qwen3_vl import Qwen3VLConfig, Qwen3VLModel

    _QWEN3_AVAILABLE = True
except ImportError:
    _QWEN3_AVAILABLE = False
    Qwen3VLConfig = None
    Qwen3VLModel = object  # Placeholder for inheritance


class ColQwen3(Qwen3VLModel):
    """
    ColQwen3 model implementation, following the architecture from the article "ColPali: Efficient Document Retrieval
    with Vision Language Models" paper. Based on the Qwen3-VL backbone.

    This model produces multi-vector embeddings (one per token) that can be used with late interaction
    retrieval methods like MaxSim scoring.

    Args:
        config (Qwen3VLConfig): The model configuration.
        mask_non_image_embeddings: Whether to ignore all token embeddings except those of the image at inference.
            Defaults to False --> Do not mask any embeddings during forward pass.
    """

    main_input_name: ClassVar[str] = "doc_input_ids"  # transformers-related

    def __init__(self, config: "Qwen3VLConfig", mask_non_image_embeddings: bool = False):
        if not _QWEN3_AVAILABLE:
            raise ImportError(
                "Qwen3VLModel not found in transformers. "
                "Please upgrade transformers: pip install --upgrade transformers>=4.53.0"
            )
        super().__init__(config=config)
        self.dim = 128
        self.custom_text_proj = nn.Linear(self.config.hidden_size, self.dim)
        self.padding_side = "left"
        self.mask_non_image_embeddings = mask_non_image_embeddings
        self.post_init()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        if not _QWEN3_AVAILABLE:
            raise ImportError(
                "Qwen3VLModel not found in transformers. "
                "Please upgrade transformers: pip install --upgrade transformers>=4.53.0"
            )
        # Pop custom arguments before passing to parent
        kwargs.pop("dim", None)  # Remove if passed, we use fixed 128
        kwargs.pop("mask_non_image_embeddings", None)
        kwargs.pop("use_cache", None)  # Handled in forward, not init

        # Handle key_mapping for checkpoint conversion if needed
        key_mapping = kwargs.pop("key_mapping", None)
        if key_mapping is None:
            parent_mapping = getattr(super(), "_checkpoint_conversion_mapping", None)
            if parent_mapping is not None:
                key_mapping = parent_mapping
        if key_mapping is not None:
            kwargs["key_mapping"] = key_mapping

        return super().from_pretrained(*args, **kwargs)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass of ColQwen3.

        Returns:
            torch.Tensor: L2-normalized embeddings of shape (batch_size, sequence_length, dim).
        """
        # Handle the custom "pixel_values" input obtained with ColQwen3Processor through unpadding
        if "pixel_values" in kwargs:
            offsets = kwargs["image_grid_thw"][:, 1] * kwargs["image_grid_thw"][:, 2]  # (batch_size,)
            kwargs["pixel_values"] = torch.cat(
                [pixel_sequence[:offset] for pixel_sequence, offset in zip(kwargs["pixel_values"], offsets)],
                dim=0,
            )

        kwargs.pop("return_dict", None)
        kwargs.pop("output_hidden_states", None)
        kwargs.pop("use_cache", None)

        last_hidden_states = (
            super()
            .forward(*args, **kwargs, use_cache=False, output_hidden_states=True, return_dict=True)
            .last_hidden_state
        )  # (batch_size, sequence_length, hidden_size)

        # Project to embedding dimension
        proj = self.custom_text_proj(last_hidden_states)  # (batch_size, sequence_length, dim)

        # L2 normalization
        proj = proj / proj.norm(dim=-1, keepdim=True)  # (batch_size, sequence_length, dim)
        proj = proj * kwargs["attention_mask"].unsqueeze(-1)  # (batch_size, sequence_length, dim)

        # Optionally mask non-image embeddings
        if "pixel_values" in kwargs and self.mask_non_image_embeddings:
            # Pools only the image embeddings
            image_mask = (kwargs["input_ids"] == self.config.image_token_id).unsqueeze(-1)
            proj = proj * image_mask

        return proj

    @property
    def patch_size(self) -> int:
        """Get the patch size from the visual encoder."""
        return self.visual.config.patch_size

    @property
    def spatial_merge_size(self) -> int:
        """Get the spatial merge size from the visual encoder."""
        return self.visual.config.spatial_merge_size
