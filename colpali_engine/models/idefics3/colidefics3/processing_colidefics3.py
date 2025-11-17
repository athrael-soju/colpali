import math
from typing import ClassVar, List, Optional, Tuple, Union

import torch
from PIL import Image
from transformers import BatchEncoding, BatchFeature, Idefics3Processor

from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor


def _resize_output_size_rescale_to_max_len(
    height: int, width: int, min_len: int = 1, max_len: Optional[int] = None
) -> Tuple[int, int]:
    """
    Get the output size of the image after resizing given a dictionary specifying the max and min sizes.

    Reused from transformers.models.idefics3.image_processing_idefics3.

    Args:
        height (`int`):
            Height of the input image.
        width (`int`):
            Width of the input image.
        min_len (`int`, *optional*, defaults to 1):
            Minimum size of the output image.
        max_len (`int`, *optional*, defaults to the maximum size of the image):
            Maximum size of the output image.

    Returns:
        The output size of the image after resizing.
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


def _resize_for_vision_encoder(height: int, width: int, vision_encoder_max_size: int) -> Tuple[int, int]:
    """
    Resize image dimensions to be multiples of `vision_encoder_max_size` while preserving the aspect ratio.

    Reused from transformers.models.idefics3.image_processing_idefics3.

    Args:
        height (`int`):
            Height of the input image.
        width (`int`):
            Width of the input image.
        vision_encoder_max_size (`int`):
            Maximum size of the output image.

    Returns:
        The output size of the image after resizing.
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
        patch_size: int,
        resize_to_max_canvas: bool = True,
    ) -> Tuple[int, int]:
        """
        Get the number of patches (n_patches_x, n_patches_y) that will be used to process an image of
        size (height, width) with the given patch size.

        Args:
            image_size: Tuple of (width, height) of the input image.
            patch_size: Size of each patch (not used for Idefics3, included for API compatibility).
            resize_to_max_canvas: Whether to resize the image to the max canvas size. Defaults to True.

        Returns:
            Tuple of (n_patches_x, n_patches_y) representing the number of patches in each dimension.

        Note:
            This method reuses the resizing logic from transformers.models.idefics3.image_processing_idefics3.
        """
        width, height = image_size
        max_image_size = self.image_processor.max_image_size["longest_edge"]

        if resize_to_max_canvas:
            # Resize to max canvas size (longest_edge strategy)
            height, width = _resize_output_size_rescale_to_max_len(height, width, max_len=max_image_size)

        # Resize to vision encoder multiples
        height, width = _resize_for_vision_encoder(height, width, max_image_size)

        # Calculate the number of patches in each dimension
        n_patches_y = math.ceil(height / max_image_size)
        n_patches_x = math.ceil(width / max_image_size)

        return n_patches_x, n_patches_y

    def get_image_mask(self, batch_images: BatchFeature) -> torch.Tensor:
        """
        Get a tensor mask that identifies the image tokens in the batch.

        Args:
            batch_images: The output of the processor containing input_ids.

        Returns:
            A boolean tensor where True indicates an image token position.
        """
        return batch_images.input_ids == self.image_token_id
