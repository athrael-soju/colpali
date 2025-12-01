from typing import ClassVar, Literal, Optional

import torch
from torch import nn

try:
    from transformers.models.qwen3_vl import Qwen3VLConfig, Qwen3VLModel

    _QWEN3_AVAILABLE = True
except ImportError:
    from transformers import AutoConfig, AutoModel

    Qwen3VLConfig = None
    Qwen3VLModel = None
    _QWEN3_AVAILABLE = False


class BiQwen3(nn.Module):
    """
    BiQwen3 is an implementation from the "ColPali: Efficient Document Retrieval with Vision Language Models" paper.
    Representations are pooled to obtain a single vector representation. Based on the Qwen3-VL backbone.

    Unlike ColQwen3 which produces multi-vector embeddings (one per token), BiQwen3 produces a single
    dense embedding per input using pooling.

    Args:
        config: The model configuration (Qwen3VLConfig or compatible).
    """

    main_input_name: ClassVar[str] = "doc_input_ids"  # transformers-related

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.padding_side = "left"

        # Initialize the base Qwen3-VL model
        if _QWEN3_AVAILABLE and Qwen3VLModel is not None:
            self.model = Qwen3VLModel(config)
        else:
            raise ImportError(
                "Qwen3VLModel not found in transformers. "
                "Please upgrade transformers to a version that supports Qwen3-VL, "
                "or use trust_remote_code=True with a compatible model."
            )

        self.post_init()

    def post_init(self):
        """Post initialization hook."""
        pass

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        **kwargs,
    ):
        """
        Load a pretrained BiQwen3 model.

        Args:
            pretrained_model_name_or_path: Path to the pretrained model or model identifier from HuggingFace.
            **kwargs: Additional arguments passed to the model loading.

        Returns:
            BiQwen3: The loaded model.
        """
        trust_remote_code = kwargs.pop("trust_remote_code", True)
        torch_dtype = kwargs.pop("torch_dtype", torch.bfloat16)
        attn_implementation = kwargs.pop("attn_implementation", "flash_attention_2")

        if _QWEN3_AVAILABLE and Qwen3VLConfig is not None:
            config = Qwen3VLConfig.from_pretrained(
                pretrained_model_name_or_path,
                trust_remote_code=trust_remote_code,
                **kwargs,
            )
            model = cls(config)

            # Load pretrained weights
            base_model = Qwen3VLModel.from_pretrained(
                pretrained_model_name_or_path,
                config=config,
                torch_dtype=torch_dtype,
                attn_implementation=attn_implementation,
                trust_remote_code=trust_remote_code,
                **kwargs,
            )
            model.model = base_model
        else:
            # Use AutoModel with trust_remote_code for compatibility
            config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path,
                trust_remote_code=trust_remote_code,
                **kwargs,
            )
            model = cls.__new__(cls)
            nn.Module.__init__(model)

            model.config = config
            model.padding_side = "left"

            model.model = AutoModel.from_pretrained(
                pretrained_model_name_or_path,
                config=config,
                torch_dtype=torch_dtype,
                attn_implementation=attn_implementation,
                trust_remote_code=trust_remote_code,
                **kwargs,
            )

        return model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        pooling_strategy: Literal["cls", "last", "mean"] = "last",
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass for BiQwen3 model.

        Args:
            input_ids: Input token IDs.
            attention_mask: Attention mask for the input.
            pixel_values: Pixel values of the images.
            image_grid_thw: Image grid dimensions (time, height, width).
            pooling_strategy: The strategy to use for pooling the hidden states.
                - "cls": Use the first token embedding.
                - "last": Use the last token embedding (default, good for left-padded inputs).
                - "mean": Mean pool over all tokens.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: Dense embeddings (batch_size, hidden_size).
        """
        # Handle the custom "pixel_values" input obtained with processor through unpadding
        if pixel_values is not None and image_grid_thw is not None:
            offsets = image_grid_thw[:, 1] * image_grid_thw[:, 2]  # (batch_size,)
            pixel_values = torch.cat(
                [pixel_sequence[:offset] for pixel_sequence, offset in zip(pixel_values, offsets)],
                dim=0,
            )

        # Remove arguments that might cause issues
        kwargs.pop("return_dict", None)
        kwargs.pop("output_hidden_states", None)
        kwargs.pop("use_cache", None)

        # Get last hidden states from the base model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
            **kwargs,
        )
        last_hidden_states = outputs.last_hidden_state  # (batch_size, sequence_length, hidden_size)

        # Get pooled representation based on strategy
        if pooling_strategy == "cls":
            # Use CLS token (first token) embedding
            pooled_output = last_hidden_states[:, 0]  # (batch_size, hidden_size)
        elif pooling_strategy == "last":
            # Use last token since we are left padding
            pooled_output = last_hidden_states[:, -1]  # (batch_size, hidden_size)
        elif pooling_strategy == "mean":
            # Mean pooling over sequence length
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1)  # (batch_size, sequence_length, 1)
                pooled_output = (last_hidden_states * mask).sum(dim=1) / mask.sum(dim=1)  # (batch_size, hidden_size)
            else:
                pooled_output = last_hidden_states.mean(dim=1)  # (batch_size, hidden_size)
        else:
            raise ValueError(f"Invalid pooling strategy: {pooling_strategy}")

        # L2 normalization
        pooled_output = pooled_output / pooled_output.norm(dim=-1, keepdim=True)

        return pooled_output

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def save_pretrained(self, save_directory: str, **kwargs):
        """Save the model to a directory."""
        self.model.save_pretrained(save_directory, **kwargs)

    @property
    def patch_size(self) -> int:
        """Get the patch size from the visual encoder."""
        if hasattr(self.model, "visual"):
            return self.model.visual.config.patch_size
        return getattr(self.config, "patch_size", 14)

    @property
    def spatial_merge_size(self) -> int:
        """Get the spatial merge size from the visual encoder."""
        if hasattr(self.model, "visual"):
            return self.model.visual.config.spatial_merge_size
        return getattr(self.config, "spatial_merge_size", 2)
