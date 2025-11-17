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
    Rescale image dimensions so that the longest side equals `max_len`,
    while preserving aspect ratio and ensuring even dimensions.
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

    height = max(height, min_len)
    width = max(width, min_len)
    return height, width


def _resize_output_size_scale_below_upper_bound(
    height: int, width: int, max_len: Optional[int] = None
) -> Tuple[int, int]:
    """
    Scale dimensions below an upper bound while preserving aspect ratio.
    """
    max_len = max(height, width) if max_len is None else max_len

    aspect_ratio = width / height
    if width >= height and width > max_len:
        width = max_len
        height = int(width / aspect_ratio)
    elif height > width and height > max_len:
        height = max_len
        width = int(height * aspect_ratio)

    height = max(height, 1)
    width = max(width, 1)
    return height, width


class ColIdefics3Processor(BaseVisualRetrieverProcessor, Idefics3Processor):
    """
    Processor for ColIdefics3.
    """

    query_augmentation_token: ClassVar[str] = "<end_of_utterance>"
    image_token: ClassVar[str] = "<image>"
    visual_prompt_prefix: ClassVar[str] = "<|im_start|>User:<image>Describe the image.<end_of_utterance>\nAssistant:"

    def __init__(self, *args, image_seq_len=64, **kwargs):
        super().__init__(*args, image_seq_len=image_seq_len, **kwargs)
        self.tokenizer.padding_side = "left"

    def process_images(
        self,
        images: List[Image.Image],
    ) -> Union[BatchFeature, BatchEncoding]:
        """
        Process images for ColIdefics3.

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
        Process texts for ColIdefics3.

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
        patch_size: int = 14,
        resize_to_max_len: bool = True,
    ) -> Tuple[int, int]:
        """
        Get the number of patches (n_patches_x, n_patches_y) that will be used to process an image of
        size (width, height).

        For Idefics3 with image splitting enabled, this returns the patch grid dimensions for the
        combined split tiles (excluding the global tile). Each tile has sqrt(image_seq_len) x sqrt(image_seq_len)
        patches.

        Args:
            image_size: Tuple of (width, height) of the input image.
            patch_size: The SigLIP patch size (typically 14). This parameter is kept for API consistency
                but is not directly used in Idefics3. Instead, image_seq_len determines patches per tile.
            resize_to_max_len: Whether to resize to longest_edge first. Set to False if the image
                is already at the target resolution.

        Returns:
            Tuple of (n_patches_x, n_patches_y).
        """
        width, height = image_size

        if resize_to_max_len:
            # Get the target resolution from image processor
            longest_edge = self.image_processor.size.get("longest_edge", max(width, height))

            # Step 1: Rescale to the target longest edge
            height, width = _resize_output_size_rescale_to_max_len(height, width, max_len=longest_edge)

            # Step 2: Ensure we don't exceed the absolute max
            height, width = _resize_output_size_scale_below_upper_bound(height, width, max_len=MAX_IMAGE_SIZE)

        # Get max_image_size (tile size for splitting)
        max_image_size = self.image_processor.max_image_size.get("longest_edge", 384)

        # Get patches per tile dimension (sqrt of image_seq_len)
        patches_per_tile = int(math.sqrt(self.image_seq_len))

        # Check if image splitting is enabled and needed
        do_image_splitting = getattr(self.image_processor, "do_image_splitting", False)

        if do_image_splitting and (height > max_image_size or width > max_image_size):
            # Calculate number of tiles needed
            num_tiles_y = math.ceil(height / max_image_size)
            num_tiles_x = math.ceil(width / max_image_size)

            # Total patches = tiles * patches_per_tile
            n_patches_x = num_tiles_x * patches_per_tile
            n_patches_y = num_tiles_y * patches_per_tile
        else:
            # No splitting, just one tile
            n_patches_x = patches_per_tile
            n_patches_y = patches_per_tile

        return n_patches_x, n_patches_y

    def get_image_mask(self, batch_images: BatchFeature) -> torch.Tensor:
        """
        Get a tensor mask that identifies the image tokens in the batch.

        For Idefics3 with image splitting, this returns a mask for the split tiles only,
        excluding the global tile. This is necessary for proper spatial interpretation
        of similarity maps.

        Args:
            batch_images: BatchFeature containing the processed images with input_ids.

        Returns:
            Boolean tensor of shape (batch_size, sequence_length) where True indicates
            image token positions for split tiles only.
        """
        # Get the image token id from the tokenizer
        image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)

        # Create mask for all image tokens
        all_image_mask = batch_images.input_ids == image_token_id

        # Check if we need to exclude the global tile
        # The global tile is the last tile in the sequence (appended after split tiles)
        do_image_splitting = getattr(self.image_processor, "do_image_splitting", False)

        if not do_image_splitting:
            # No splitting, use all image tokens
            return all_image_mask

        # Count total image tokens per batch item
        batch_size = batch_images.input_ids.shape[0]
        result_mask = all_image_mask.clone()

        for batch_idx in range(batch_size):
            num_image_tokens = all_image_mask[batch_idx].sum().item()
            num_tokens_per_tile = self.image_seq_len

            # Check if there are enough tokens to have a global tile
            if num_image_tokens > num_tokens_per_tile:
                num_tiles = num_image_tokens // num_tokens_per_tile

                # If we have multiple tiles, the last one is the global tile
                if num_tiles > 1:
                    # Find the positions of image tokens
                    image_positions = torch.where(all_image_mask[batch_idx])[0]

                    # The last image_seq_len tokens belong to the global tile
                    # We want to exclude these from the mask
                    global_tile_start = len(image_positions) - num_tokens_per_tile
                    global_tile_positions = image_positions[global_tile_start:]

                    # Set the global tile positions to False
                    result_mask[batch_idx, global_tile_positions] = False

        return result_mask
