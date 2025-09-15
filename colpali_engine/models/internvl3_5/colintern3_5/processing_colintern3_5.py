from typing import ClassVar, List, Optional, Tuple, Union
import torch
from PIL import Image
from transformers import BatchEncoding, BatchFeature
from transformers.models.internvl import InternVLProcessor
from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor

class ColIntern3_5Processor(BaseVisualRetrieverProcessor, InternVLProcessor):  # noqa: N801
    """
    Processor for the ColIntern3.5 model.
    Combines an image processor and tokenizer to prepare inputs for ColIntern3.5, following ColPali conventions.

    Key design:
      - Right padding for tokenizer
      - Control visual detail via image size (no arbitrary 'max_patches' cap)
      - Pixel values cast to bf16
      - `downsample_ratio` is provided or set later from the model config to avoid brittle fallbacks
    """

    visual_prompt_prefix: ClassVar[str] = "<IMG_CONTEXT> Describe the image."
    query_augmentation_token: ClassVar[str] = "<|endoftext|>"

    def __init__(self, *args, **kwargs):
        # Accept explicit control from trainer/runner
        self._initial_downsample_ratio = kwargs.pop("downsample_ratio", None)
        self._initial_max_image_size = kwargs.pop("max_image_size", 448)
        super().__init__(*args, **kwargs)

        # Ensure text tokenizer pads on the right
        self.tokenizer.padding_side = "right"

        # Downsample ratio: prefer explicit value; otherwise conservative default
        self.downsample_ratio = self._initial_downsample_ratio if self._initial_downsample_ratio is not None else 0.5

        # Expose image tokens if available
        self.image_token = getattr(self.tokenizer, 'context_image_token', '<IMG_CONTEXT>')
        self.start_image_token = getattr(self.tokenizer, 'start_image_token', '')
        self.end_image_token = getattr(self.tokenizer, 'end_image_token', '')
        if hasattr(self.tokenizer, "context_image_token_id"):
            self.image_token_id = self.tokenizer.context_image_token_id
        elif hasattr(self.tokenizer, "image_token_id"):
            self.image_token_id = self.tokenizer.image_token_id

        # Standardize image size for document pages; do NOT impose arbitrary patch caps
        if hasattr(self, "image_processor") and hasattr(self.image_processor, "size"):
            if isinstance(self.image_processor.size, dict):
                self.image_processor.size = {"height": self._initial_max_image_size, "width": self._initial_max_image_size}
            else:
                self.image_processor.size = self._initial_max_image_size
        if hasattr(self, "image_processor") and hasattr(self.image_processor, "crop_to_patches"):
            self.image_processor.crop_to_patches = True

    @property
    def query_augmentation_token(self) -> str:
        """Return the query augmentation token."""
        return self.tokenizer.pad_token

    @classmethod
    def from_pretrained(cls, *args, device_map: Optional[str] = None, **kwargs):
        """
        Optional kwargs:
          - max_image_size: int (e.g., 448). If provided, we resize the longer side to this value.
          - downsample_ratio: float. If provided, used directly; otherwise trainer should assign from model.config.
        """
        return super().from_pretrained(*args, device_map=device_map, **kwargs)

    def process_images(self, images: List[Image.Image]) -> BatchEncoding:
        """Process images for the model using the InternVL processor."""
        images = [image.convert("RGB") for image in images]
        placeholder = self.visual_prompt_prefix
        batch = self(
            text=[placeholder] * len(images),
            images=images,
            padding="longest",
            return_tensors="pt",
        )
        if "pixel_values" in batch:
            batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16)
        return batch

    def process_texts(self, texts: List[str]) -> Union[BatchFeature, BatchEncoding]:
        """Process a batch of text queries for input to ColIntern3.5."""
        return self(text=texts, return_tensors="pt", padding="longest")

    # Alias
    def process_queries(self, queries: List[str]) -> Union[BatchFeature, BatchEncoding]:
        return self.process_texts(queries)

    def score(
        self,
        qs: List[torch.Tensor],
        ps: List[torch.Tensor],
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute the MaxSim score (ColBERT-like) for the given multi-vector query and passage embeddings."""
        return self.score_multi_vector(qs, ps, device=device, **kwargs)

    def score_multi_vector(self, qs, ps, device: Optional[Union[str, torch.device]] = None) -> torch.Tensor:
        """Compute ColBERT-style MaxSim score (delegates to BaseVisualRetrieverProcessor)."""
        return super().score_multi_vector(qs, ps, device=device)

    def get_n_patches(self, image_size: Tuple[int, int], spatial_merge_size: int) -> Tuple[int, int]:
        """Compute the number of patch tokens (n_patches_x, n_patches_y) for an image of given (height, width)."""
        patch_size = getattr(self.image_processor, "patch_size", 14)
        if isinstance(patch_size, (list, tuple)):
            patch_size = patch_size[0]
        height, width = image_size
        factor = patch_size * spatial_merge_size

        # Respect the processor's intended resize target (longer side)
        if hasattr(self.image_processor, "size"):
            target = self.image_processor.size
            if isinstance(target, dict):
                max_side = max(target.get("height", 448), target.get("width", 448))
            else:
                max_side = int(target)
        else:
            max_side = 448

        if max(height, width) > max_side:
            scale = max_side / max(height, width)
            height = int(height * scale)
            width = int(width * scale)

        # Align to multiples of factor
        height_new = max(factor, (height // factor) * factor)
        width_new = max(factor, (width // factor) * factor)
        n_patches_y = height_new // patch_size // spatial_merge_size
        n_patches_x = width_new // patch_size // spatial_merge_size
        return (n_patches_x, n_patches_y)

    def get_image_mask(self, batch_images: BatchFeature) -> torch.Tensor:
        """Generate a mask indicating which positions correspond to image patch tokens."""
        return batch_images["input_ids"] == getattr(self, "image_token_id", self.tokenizer.context_image_token_id)
