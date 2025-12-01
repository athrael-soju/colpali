from typing import ClassVar, Optional

import torch
from torch import nn

try:
    from transformers.models.qwen3_vl import Qwen3VLConfig, Qwen3VLModel
except ImportError:
    # Fallback for older transformers versions - use dynamic import
    from transformers import AutoConfig, AutoModel

    Qwen3VLConfig = None
    Qwen3VLModel = None


class ColQwen3(nn.Module):
    """
    ColQwen3 model implementation, following the architecture from the article "ColPali: Efficient Document Retrieval
    with Vision Language Models" paper. Based on the Qwen3-VL backbone.

    This model produces multi-vector embeddings (one per token) that can be used with late interaction
    retrieval methods like MaxSim scoring.

    Args:
        config: The model configuration (Qwen3VLConfig or compatible).
        dim: The output embedding dimension. Defaults to 128 for consistency with other ColPali models.
            Note: TomoroAI's implementation uses 320-dim, but 128 is standard for ColPali models.
        mask_non_image_embeddings: Whether to ignore all token embeddings except those of the image at inference.
            Defaults to False --> Do not mask any embeddings during forward pass.
    """

    main_input_name: ClassVar[str] = "doc_input_ids"  # transformers-related

    def __init__(
        self,
        config,
        dim: int = 128,
        mask_non_image_embeddings: bool = False,
    ):
        super().__init__()
        self.config = config
        self.dim = dim
        self.mask_non_image_embeddings = mask_non_image_embeddings
        self.padding_side = "left"

        # Initialize the base Qwen3-VL model
        if Qwen3VLModel is not None:
            self.model = Qwen3VLModel(config)
        else:
            raise ImportError(
                "Qwen3VLModel not found in transformers. "
                "Please upgrade transformers to a version that supports Qwen3-VL, "
                "or use trust_remote_code=True with a compatible model."
            )

        # Custom projection layer for ColBERT-style embeddings
        self.custom_text_proj = nn.Linear(self.config.hidden_size, self.dim)

        self.post_init()

    def post_init(self):
        """Initialize weights for the projection layer."""
        self.custom_text_proj.weight.data.normal_(mean=0.0, std=0.02)
        if self.custom_text_proj.bias is not None:
            self.custom_text_proj.bias.data.zero_()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        dim: int = 128,
        mask_non_image_embeddings: bool = False,
        **kwargs,
    ):
        """
        Load a pretrained ColQwen3 model.

        Args:
            pretrained_model_name_or_path: Path to the pretrained model or model identifier from HuggingFace.
            dim: The output embedding dimension.
            mask_non_image_embeddings: Whether to mask non-image embeddings at inference.
            **kwargs: Additional arguments passed to the model loading.

        Returns:
            ColQwen3: The loaded model.
        """
        trust_remote_code = kwargs.pop("trust_remote_code", True)
        torch_dtype = kwargs.pop("torch_dtype", torch.bfloat16)
        attn_implementation = kwargs.pop("attn_implementation", "flash_attention_2")

        if Qwen3VLConfig is not None:
            config = Qwen3VLConfig.from_pretrained(
                pretrained_model_name_or_path,
                trust_remote_code=trust_remote_code,
                **kwargs,
            )
            model = cls(config, dim=dim, mask_non_image_embeddings=mask_non_image_embeddings)

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
            model.dim = dim
            model.mask_non_image_embeddings = mask_non_image_embeddings
            model.padding_side = "left"

            model.model = AutoModel.from_pretrained(
                pretrained_model_name_or_path,
                config=config,
                torch_dtype=torch_dtype,
                attn_implementation=attn_implementation,
                trust_remote_code=trust_remote_code,
                **kwargs,
            )

            model.custom_text_proj = nn.Linear(config.hidden_size, dim)
            model.post_init()

        return model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass of ColQwen3.

        Args:
            input_ids: Input token IDs.
            attention_mask: Attention mask for the input.
            pixel_values: Pixel values of the images.
            image_grid_thw: Image grid dimensions (time, height, width).
            **kwargs: Additional arguments.

        Returns:
            torch.Tensor: L2-normalized embeddings of shape (batch_size, sequence_length, dim).
        """
        # Handle the custom "pixel_values" input obtained with ColQwen3Processor through unpadding
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

        # Project to embedding dimension
        proj = self.custom_text_proj(last_hidden_states)  # (batch_size, sequence_length, dim)

        # L2 normalization
        proj = proj / proj.norm(dim=-1, keepdim=True)  # (batch_size, sequence_length, dim)

        # Apply attention mask
        if attention_mask is not None:
            proj = proj * attention_mask.unsqueeze(-1)  # (batch_size, sequence_length, dim)

        # Optionally mask non-image embeddings
        if pixel_values is not None and self.mask_non_image_embeddings:
            # Pools only the image embeddings
            image_token_id = getattr(self.config, "image_token_id", None)
            if image_token_id is not None and input_ids is not None:
                image_mask = (input_ids == image_token_id).unsqueeze(-1)
                proj = proj * image_mask

        return proj

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def save_pretrained(self, save_directory: str, **kwargs):
        """Save the model to a directory."""
        self.model.save_pretrained(save_directory, **kwargs)
        # Save the projection layer separately
        torch.save(
            {"custom_text_proj": self.custom_text_proj.state_dict(), "dim": self.dim},
            f"{save_directory}/colqwen3_projection.pt",
        )

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
