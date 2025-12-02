from typing import ClassVar, Literal

import torch
from torch import nn

try:
    from transformers.models.qwen3_vl import Qwen3VLConfig, Qwen3VLModel

    _QWEN3_AVAILABLE = True
except ImportError:
    _QWEN3_AVAILABLE = False
    Qwen3VLConfig = None
    Qwen3VLModel = object  # Placeholder for inheritance


class BiQwen3(Qwen3VLModel):
    """
    BiQwen3 is an implementation from the "ColPali: Efficient Document Retrieval with Vision Language Models" paper.
    Representations are pooled to obtain a single vector representation. Based on the Qwen3-VL backbone.

    Unlike ColQwen3 which produces multi-vector embeddings (one per token), BiQwen3 produces a single
    dense embedding per input using pooling.
    """

    main_input_name: ClassVar[str] = "doc_input_ids"  # transformers-related

    def __init__(self, config: "Qwen3VLConfig"):
        if not _QWEN3_AVAILABLE:
            raise ImportError(
                "Qwen3VLModel not found in transformers. "
                "Please upgrade transformers: pip install --upgrade transformers>=4.53.0"
            )
        super().__init__(config=config)
        self.padding_side = "left"
        self.post_init()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        if not _QWEN3_AVAILABLE:
            raise ImportError(
                "Qwen3VLModel not found in transformers. "
                "Please upgrade transformers: pip install --upgrade transformers>=4.53.0"
            )
        # Handle key_mapping for checkpoint conversion if needed
        key_mapping = kwargs.pop("key_mapping", None)
        if key_mapping is None:
            parent_mapping = getattr(super(), "_checkpoint_conversion_mapping", None)
            if parent_mapping is not None:
                key_mapping = parent_mapping
        if key_mapping is not None:
            kwargs["key_mapping"] = key_mapping

        return super().from_pretrained(*args, **kwargs)

    def forward(
        self,
        pooling_strategy: Literal["cls", "last", "mean"] = "last",
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass for BiQwen3 model.

        Args:
            pooling_strategy: The strategy to use for pooling the hidden states.
                - "cls": Use the first token embedding.
                - "last": Use the last token embedding (default, good for left-padded inputs).
                - "mean": Mean pool over all tokens.
            *args: Variable length argument list.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: Dense embeddings (batch_size, hidden_size).
        """
        # Handle the custom "pixel_values" input through unpadding
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

        # Get pooled representation based on strategy
        if pooling_strategy == "cls":
            # Use CLS token (first token) embedding
            pooled_output = last_hidden_states[:, 0]  # (batch_size, hidden_size)
        elif pooling_strategy == "last":
            # Use last token since we are left padding
            pooled_output = last_hidden_states[:, -1]  # (batch_size, hidden_size)
        elif pooling_strategy == "mean":
            # Mean pooling over sequence length
            mask = kwargs["attention_mask"].unsqueeze(-1)  # (batch_size, sequence_length, 1)
            pooled_output = (last_hidden_states * mask).sum(dim=1) / mask.sum(dim=1)  # (batch_size, hidden_size)
        else:
            raise ValueError(f"Invalid pooling strategy: {pooling_strategy}")

        # L2 normalization
        pooled_output = pooled_output / pooled_output.norm(dim=-1, keepdim=True)

        return pooled_output

    @property
    def patch_size(self) -> int:
        """Get the patch size from the visual encoder."""
        return self.visual.config.patch_size

    @property
    def spatial_merge_size(self) -> int:
        """Get the spatial merge size from the visual encoder."""
        return self.visual.config.spatial_merge_size
