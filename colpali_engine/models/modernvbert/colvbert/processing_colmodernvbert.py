import math
from typing import ClassVar, List, Optional, Tuple, Union

import torch
from PIL import Image
from transformers import BatchEncoding, BatchFeature, Idefics3Processor

from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor


MAX_IMAGE_SIZE = 4096  # 4k resolution as absolute maximum


def _resize_output_size_rescale_to_max_len(
    height: int, width: int, min_len: int = 1, max_len: Optional[int] = None
) -> Tuple[int, int]:
    """
    Rescale the longest edge to max_len while maintaining aspect ratio.

    Args:
        height: Height of the input image.
        width: Width of the input image.
        min_len: Minimum size of the output image.
        max_len: Maximum size of the output image (longest edge).

    Returns:
        The output size (height, width) after resizing.
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
    Scale image so that the longest dimension doesn't exceed max_len.

    Args:
        height: Height of the input image.
        width: Width of the input image.
        max_len: Maximum dimension allowed.

    Returns:
        The output size (height, width) after resizing.
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

    def get_image_mask(self, batch_images: BatchFeature) -> torch.Tensor:
        """
        Get a tensor mask that identifies the image tokens in the batch.

        Args:
            batch_images: The batch of processed images.

        Returns:
            A boolean tensor of shape (batch_size, sequence_length) where True indicates image tokens.
        """
        image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)
        return batch_images.input_ids == image_token_id

    def get_n_patches(
        self,
        image_size: Tuple[int, int],
        patch_size: int,
        resize_to_max_len: bool = True,
        include_global_view: bool = True,
    ) -> Tuple[int, int]:
        """
        Get the number of patches (n_patches_x, n_patches_y) that will be used to process an image.

        This method calculates how many visual tokens the image will produce based on the
        image processor's configuration (do_image_splitting, max_image_size, size) and
        the image_seq_len parameter.

        For Idefics3-style models, images are split into sub-images, each producing
        image_seq_len tokens arranged in a square grid. When image splitting is enabled,
        an additional global view is prepended, adding image_seq_len tokens.

        Note: When include_global_view is True (default), the returned grid dimensions
        account for the global view by adding one extra row of tokens, which may not
        perfectly match the spatial layout of the original image but ensures all tokens
        are accounted for.

        Args:
            image_size: Tuple of (width, height) of the input image.
            patch_size: The patch size used by the vision encoder (unused for Idefics3 as
                        patch calculation is based on image splitting, not vision encoder patches).
            resize_to_max_len: If True, resize the longest edge to size["longest_edge"] before
                               calculating patches. If False, use the original image size.
            include_global_view: If True, account for the global view tokens added by Idefics3
                                when image splitting is enabled. Defaults to True.

        Returns:
            Tuple of (n_patches_x, n_patches_y) representing the number of visual tokens in each dimension.
        """
        width, height = image_size

        # Get image processor configuration
        do_image_splitting = getattr(self.image_processor, "do_image_splitting", True)
        max_image_size = getattr(self.image_processor, "max_image_size", {"longest_edge": 364})
        size = getattr(self.image_processor, "size", {"longest_edge": 1456})

        # Default to 1x1 if no splitting
        num_rows = 1
        num_cols = 1

        if do_image_splitting:
            # Apply resizing logic
            if resize_to_max_len:
                height, width = _resize_output_size_rescale_to_max_len(height, width, max_len=size["longest_edge"])
                height, width = _resize_output_size_scale_below_upper_bound(height, width, max_len=MAX_IMAGE_SIZE)
            else:
                # Just ensure we're below the absolute maximum
                height, width = _resize_output_size_scale_below_upper_bound(height, width, max_len=MAX_IMAGE_SIZE)

            # Calculate the number of splits based on max_image_size
            max_height = max_image_size.get("longest_edge", 364)
            max_width = max_image_size.get("longest_edge", 364)

            # Round up dimensions to multiples of max_image_size for splitting calculation
            aspect_ratio = width / height

            if width >= height:
                resized_width = math.ceil(width / max_width) * max_width
                resized_height = int(resized_width / aspect_ratio)
                resized_height = math.ceil(height / max_height) * max_height
            else:
                resized_height = math.ceil(height / max_height) * max_height
                resized_width = int(resized_height * aspect_ratio)
                resized_width = math.ceil(width / max_width) * max_width

            # Calculate number of splits
            if resized_height > max_height or resized_width > max_width:
                num_rows = math.ceil(resized_height / max_height)
                num_cols = math.ceil(resized_width / max_width)

        # Calculate tokens per split dimension
        # Each split produces image_seq_len tokens arranged in a square grid
        tokens_per_split_side = int(math.sqrt(self.image_seq_len))

        # Total number of tokens in each dimension
        n_patches_x = num_cols * tokens_per_split_side
        n_patches_y = num_rows * tokens_per_split_side

        # Account for global view if enabled
        # Idefics3 prepends a global view when image splitting is enabled
        # The global view adds image_seq_len tokens which we arrange as additional rows
        if include_global_view and do_image_splitting and (num_rows > 1 or num_cols > 1):
            # Global view tokens are arranged as additional rows at the top
            # Number of additional rows = image_seq_len / n_patches_x
            global_view_rows = self.image_seq_len // n_patches_x
            n_patches_y += global_view_rows

        return n_patches_x, n_patches_y
