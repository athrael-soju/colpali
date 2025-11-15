from typing import ClassVar, List, Optional, Tuple, Union

import torch
from PIL import Image
from transformers import BatchEncoding, BatchFeature, Idefics3Processor

from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor


class ColModernVBertProcessor(BaseVisualRetrieverProcessor, Idefics3Processor):
    """
    Processor for ColIdefics3.
    """

    query_augmentation_token: ClassVar[str] = "<end_of_utterance>"
    image_token: ClassVar[str] = "<image>"
    visual_prompt_prefix: ClassVar[str] = (
        "<|begin_of_text|>User:<image>Describe the image.<end_of_utterance>\nAssistant:"
    )

    def __init__(self, *args, image_seq_len=64, **kwargs):
        super().__init__(*args, image_seq_len=image_seq_len, **kwargs)
        self.tokenizer.padding_side = "left"

    def process_images(
        self,
        images: List[Image.Image],
    ) -> Union[BatchFeature, BatchEncoding]:
        """
        Process images for ColModernVBert.

        Args:
            images: List of PIL images.
        """
        images = [image.convert("RGB") for image in images]

        batch_doc = self(
            text=[self.visual_prompt_prefix] * len(images),
            images=images,
            padding="longest",
            return_tensors="pt",
        )
        return batch_doc

    def process_texts(self, texts: List[str]) -> Union[BatchFeature, BatchEncoding]:
        """
        Process texts for ColModernVBert.

        Args:
            texts: List of input texts.

        Returns:
            Union[BatchFeature, BatchEncoding]: Processed texts.
        """
        return self(
            text=texts,
            return_tensors="pt",
            padding="longest",
        )

    def score(
        self,
        qs: List[torch.Tensor],
        ps: List[torch.Tensor],
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute the MaxSim score (ColBERT-like) for the given multi-vector query and passage embeddings.
        """
        return self.score_multi_vector(qs, ps, device=device, **kwargs)

    def get_n_patches(
        self,
        image_size: Tuple[int, int],
        patch_size: int,
    ) -> Tuple[int, int]:
        """
        Get the number of patches (n_patches_x, n_patches_y) for ModernVBERT models.

        ModernVBERT uses an Idefics3Processor with a SigLIP vision encoder. Images are
        split into sub-images of size `max_image_size`, which are then divided into
        patches of size `patch_size`.

        Args:
            image_size: Original image size (width, height) - not used as images
                       are resized to fixed sub-image sizes.
            patch_size: Size of each patch (typically 16 for SigLIP-based models).

        Returns:
            Tuple of (n_patches_x, n_patches_y) representing the number of patches in each dimension.
        """
        # For ModernVBERT (using Idefics3Processor), get the sub-image size from configuration
        # Images are split into sub-images of this size before being divided into patches
        if hasattr(self.image_processor, "max_image_size"):
            # max_image_size defines the size of each sub-image
            if isinstance(self.image_processor.max_image_size, dict):
                sub_image_size = self.image_processor.max_image_size.get("longest_edge", 364)
            else:
                sub_image_size = self.image_processor.max_image_size
        elif hasattr(self.image_processor, "size") and isinstance(self.image_processor.size, dict):
            # Fallback: check if size is defined
            size_value = self.image_processor.size.get("longest_edge", 364)
            # If size is the overall longest_edge (e.g., 1456), divide by default factor of 4
            # to get sub-image size (364). Otherwise use as-is.
            sub_image_size = size_value if size_value < 1000 else 364
        else:
            # Default sub-image size for Idefics3-based processors (364x364)
            sub_image_size = 364

        # Calculate number of patches in each dimension
        # For Idefics3-based processors, sub-images are square
        n_patches_x = sub_image_size // patch_size
        n_patches_y = sub_image_size // patch_size

        return n_patches_x, n_patches_y
