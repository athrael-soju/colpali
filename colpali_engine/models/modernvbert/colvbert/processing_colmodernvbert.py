import math
from typing import ClassVar, List, Optional, Tuple, Union

import torch
from PIL import Image
from transformers import BatchEncoding, BatchFeature, Idefics3Processor

from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor

MAX_IMAGE_SIZE = 4096


def _resize_output_size_rescale_to_max_len(
    height: int, width: int, min_len: int = 1, max_len: Optional[int] = None
) -> Tuple[int, int]:
    """
    Get the output size of the image after resizing given a dictionary specifying the max and min sizes.
    """
    max_len = max(height, width) if max_len is None else max_len
    aspect_ratio = width / height

    if width >= height:
        width = max_len
        height = int(width / aspect_ratio)
        if height % 2 != 0:
            height += 1
    elif height > width:
        height = max_len
        width = int(height * aspect_ratio)
        if width % 2 != 0:
            width += 1

    # Avoid resizing to a size smaller than min_len
    height = max(height, min_len)
    width = max(width, min_len)
    return height, width


def _resize_output_size_scale_below_upper_bound(
    height: int, width: int, max_len: Optional[int] = None
) -> Tuple[int, int]:
    """
    Get the output size of the image after resizing given a dictionary specifying the max and min sizes.
    """
    max_len = max(height, width) if max_len is None else max_len

    aspect_ratio = width / height
    if width >= height and width > max_len:
        width = max_len
        height = int(width / aspect_ratio)
    elif height > width and height > max_len:
        height = max_len
        width = int(height * aspect_ratio)

    # Avoid resizing to a size smaller than 1
    height = max(height, 1)
    width = max(width, 1)
    return height, width


def _get_resize_output_image_size(
    height: int,
    width: int,
    resolution_max_side: int,
) -> Tuple[int, int]:
    """
    Get the output size of the image after resizing given a dictionary specifying the max and min sizes.
    """
    # Find the output size, when rescaling the longest edge to max_len and preserving the aspect ratio
    height, width = _resize_output_size_rescale_to_max_len(height, width, max_len=resolution_max_side)
    # Find the output size when scaling the image to be below the MAX_IMAGE_SIZE
    height, width = _resize_output_size_scale_below_upper_bound(height, width, max_len=MAX_IMAGE_SIZE)
    return height, width


def _resize_for_vision_encoder(
    height: int,
    width: int,
    vision_encoder_max_size: int,
) -> Tuple[int, int]:
    """
    Get the dimensions after resizing for the vision encoder (make dimensions multiples of vision_encoder_max_size).
    """
    aspect_ratio = width / height
    if width >= height:
        width = math.ceil(width / vision_encoder_max_size) * vision_encoder_max_size
        height = int(width / aspect_ratio)
        height = math.ceil(height / vision_encoder_max_size) * vision_encoder_max_size
    elif height > width:
        height = math.ceil(height / vision_encoder_max_size) * vision_encoder_max_size
        width = int(height * aspect_ratio)
        width = math.ceil(width / vision_encoder_max_size) * vision_encoder_max_size
    return height, width


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
        Get the number of patches (n_patches_x, n_patches_y) that will be used to process an image of
        size (height, width) with the given patch size.

        The calculation follows the Idefics3 preprocessing pipeline:
        1. If do_resize is True, resize the image based on the `size` parameter
        2. If do_image_splitting is True, further resize to multiples of `max_image_size`
        3. Calculate the number of patches based on the final dimensions and patch_size

        Args:
            image_size: Tuple of (width, height) of the original image.
            patch_size: The size of each patch (e.g., 14 for SigLIP).

        Returns:
            Tuple of (n_patches_x, n_patches_y) representing the number of patches in x and y dimensions.
        """
        width, height = image_size

        # Step 1: Apply initial resize based on the processor's size configuration
        if self.image_processor.do_resize:
            size_config = self.image_processor.size
            if "longest_edge" in size_config:
                # Resize so that the longest edge equals size["longest_edge"], preserving aspect ratio
                height, width = _get_resize_output_image_size(height, width, size_config["longest_edge"])
            elif "height" in size_config and "width" in size_config:
                # Resize to exact dimensions (no aspect ratio preservation)
                height = size_config["height"]
                width = size_config["width"]

        # Step 2: If image splitting is enabled, resize to multiples of max_image_size
        if self.image_processor.do_image_splitting:
            max_image_size = self.image_processor.max_image_size
            if "longest_edge" in max_image_size:
                vision_encoder_max_size = max_image_size["longest_edge"]
            else:
                # Fallback to height/width if longest_edge is not specified
                vision_encoder_max_size = max_image_size.get("height", max_image_size.get("width", 364))
            height, width = _resize_for_vision_encoder(height, width, vision_encoder_max_size)

        # Step 3: Calculate the number of patches
        n_patches_x = width // patch_size
        n_patches_y = height // patch_size

        return n_patches_x, n_patches_y
