from typing import ClassVar, List, Optional, Tuple, Union

import torch
from PIL import Image
from transformers import AutoProcessor, BatchEncoding, BatchFeature

from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor

# Try to import Qwen3-VL specific processor, fallback to AutoProcessor
try:
    from transformers.models.qwen3_vl import Qwen3VLProcessor

    _QWEN3_PROCESSOR_AVAILABLE = True
except ImportError:
    _QWEN3_PROCESSOR_AVAILABLE = False
    Qwen3VLProcessor = None

# Try to import smart_resize for image processing
try:
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
except ImportError:
    # Fallback implementation
    def smart_resize(width: int, height: int, factor: int, min_pixels: int, max_pixels: int) -> Tuple[int, int]:
        """Resize image dimensions to be divisible by factor while respecting pixel limits."""
        aspect_ratio = width / height

        # Calculate target pixels
        current_pixels = width * height
        if current_pixels > max_pixels:
            scale = (max_pixels / current_pixels) ** 0.5
            width = int(width * scale)
            height = int(height * scale)
        elif current_pixels < min_pixels:
            scale = (min_pixels / current_pixels) ** 0.5
            width = int(width * scale)
            height = int(height * scale)

        # Make divisible by factor
        width = (width // factor) * factor
        height = (height // factor) * factor

        return height, width


class ColQwen3Processor(BaseVisualRetrieverProcessor):
    """
    Processor for ColQwen3.

    This processor handles the conversion of images and text into the format expected by ColQwen3.

    Args:
        *args: Variable length argument list to be passed to the parent processor class.
        max_num_visual_tokens: The maximum number of visual tokens that can be processed by the model.
            Defaults to 1280 (as recommended by TomoroAI's ColQwen3).
        **kwargs: Arbitrary keyword arguments to be passed to the parent processor class.
    """

    visual_prompt_prefix: ClassVar[str] = (
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe the image.<|im_end|><|endoftext|>"
    )
    query_augmentation_token: ClassVar[str] = "<|endoftext|>"
    image_token: ClassVar[str] = "<|image_pad|>"

    def __init__(
        self,
        processor,
        max_num_visual_tokens: int = 1280,
    ):
        """
        Initialize the ColQwen3Processor.

        Args:
            processor: The underlying processor (Qwen3VLProcessor or compatible).
            max_num_visual_tokens: Maximum number of visual tokens per image.
        """
        self.processor = processor
        self.max_num_visual_tokens = max_num_visual_tokens

        # Configure tokenizer
        if hasattr(self.processor, "tokenizer"):
            self.processor.tokenizer.padding_side = "left"

        # Configure image processor
        if hasattr(self.processor, "image_processor"):
            # Set max pixels based on visual token budget
            # Each patch is 28x28 pixels in Qwen3-VL
            self.processor.image_processor.max_pixels = max_num_visual_tokens * 28 * 28
            if hasattr(self.processor.image_processor, "size") and isinstance(
                self.processor.image_processor.size, dict
            ):
                self.processor.image_processor.size["longest_edge"] = self.processor.image_processor.max_pixels

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        max_num_visual_tokens: int = 1280,
        **kwargs,
    ):
        """
        Load a pretrained ColQwen3Processor.

        Args:
            pretrained_model_name_or_path: Path to the pretrained processor or model identifier from HuggingFace.
            max_num_visual_tokens: Maximum number of visual tokens per image.
            **kwargs: Additional arguments passed to the processor loading.

        Returns:
            ColQwen3Processor: The loaded processor.
        """
        trust_remote_code = kwargs.pop("trust_remote_code", True)

        if _QWEN3_PROCESSOR_AVAILABLE:
            processor = Qwen3VLProcessor.from_pretrained(
                pretrained_model_name_or_path,
                trust_remote_code=trust_remote_code,
                **kwargs,
            )
        else:
            processor = AutoProcessor.from_pretrained(
                pretrained_model_name_or_path,
                trust_remote_code=trust_remote_code,
                **kwargs,
            )

        return cls(processor=processor, max_num_visual_tokens=max_num_visual_tokens)

    def __call__(self, *args, **kwargs):
        """Delegate to the underlying processor."""
        return self.processor(*args, **kwargs)

    @property
    def tokenizer(self):
        """Get the tokenizer from the underlying processor."""
        return self.processor.tokenizer

    @property
    def image_processor(self):
        """Get the image processor from the underlying processor."""
        return self.processor.image_processor

    @property
    def image_token_id(self) -> int:
        """Get the image token ID."""
        if hasattr(self.processor, "image_token_id"):
            return self.processor.image_token_id
        return self.tokenizer.convert_tokens_to_ids(self.image_token)

    def process_images(
        self,
        images: List[Image.Image],
    ) -> Union[BatchFeature, BatchEncoding]:
        """
        Process images for ColQwen3.

        Args:
            images: List of PIL images.

        Returns:
            BatchFeature or BatchEncoding containing the processed images.
        """
        # Convert images to RGB
        images = [image.convert("RGB") for image in images]

        # Process images with visual prompt
        batch_doc = self.processor(
            text=[self.visual_prompt_prefix] * len(images),
            images=images,
            padding="longest",
            return_tensors="pt",
        )

        # Adjust pixel values for DDP compatibility
        if "image_grid_thw" in batch_doc:
            offsets = batch_doc["image_grid_thw"][:, 1] * batch_doc["image_grid_thw"][:, 2]  # (batch_size,)

            # Split the pixel_values tensor into a list of tensors, one per image
            pixel_values = list(torch.split(batch_doc["pixel_values"], offsets.tolist()))

            # Pad the list of pixel_value tensors to the same length along the sequence dimension
            batch_doc["pixel_values"] = torch.nn.utils.rnn.pad_sequence(pixel_values, batch_first=True)

        return batch_doc

    def process_queries(
        self,
        queries: List[str],
        max_length: int = 256,
        suffix: Optional[str] = None,
    ) -> Union[BatchFeature, BatchEncoding]:
        """
        Process text queries for ColQwen3.

        Queries are augmented with special tokens for late interaction retrieval.

        Args:
            queries: List of query strings.
            max_length: Maximum sequence length.
            suffix: Optional suffix to append to queries.

        Returns:
            BatchFeature or BatchEncoding containing the processed queries.
        """
        if suffix is None:
            suffix = self.query_augmentation_token * 10  # Pad with augmentation tokens

        augmented_queries = [q + suffix for q in queries]

        return self.processor(
            text=augmented_queries,
            padding="longest",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    def process_texts(
        self,
        texts: List[str],
    ) -> Union[BatchFeature, BatchEncoding]:
        """
        Process texts for ColQwen3.

        Args:
            texts: List of input texts.

        Returns:
            BatchFeature or BatchEncoding containing the processed texts.
        """
        return self.processor(
            text=texts,
            padding="longest",
            return_tensors="pt",
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

        Args:
            qs: List of query embeddings, each of shape (num_query_tokens, dim).
            ps: List of passage embeddings, each of shape (num_passage_tokens, dim).
            device: Device to perform computation on.
            **kwargs: Additional arguments.

        Returns:
            torch.Tensor: Similarity scores of shape (num_queries, num_passages).
        """
        return self.score_multi_vector(qs, ps, device=device, **kwargs)

    def get_n_patches(
        self,
        image_size: Tuple[int, int],
        spatial_merge_size: int,
    ) -> Tuple[int, int]:
        """
        Get the number of patches (n_patches_x, n_patches_y) that will be used to process an image of
        size (height, width) with the given patch size.

        Args:
            image_size: Tuple of (width, height) of the image.
            spatial_merge_size: Number of patches that will be merged spatially.

        Returns:
            Tuple of (n_patches_x, n_patches_y).
        """
        patch_size = self.image_processor.patch_size
        merge_size = getattr(self.image_processor, "merge_size", 2)

        height_new, width_new = smart_resize(
            width=image_size[0],
            height=image_size[1],
            factor=patch_size * merge_size,
            min_pixels=self.image_processor.size.get("shortest_edge", 256),
            max_pixels=self.image_processor.size.get("longest_edge", self.max_num_visual_tokens * 28 * 28),
        )

        n_patches_x = width_new // patch_size // spatial_merge_size
        n_patches_y = height_new // patch_size // spatial_merge_size

        return n_patches_x, n_patches_y

    def get_image_mask(self, batch_images: BatchFeature) -> torch.Tensor:
        """
        Get a mask indicating which tokens correspond to image content.

        Args:
            batch_images: Processed batch of images.

        Returns:
            torch.Tensor: Boolean mask of shape (batch_size, sequence_length).
        """
        return batch_images.input_ids == self.image_token_id

    def save_pretrained(self, save_directory: str, **kwargs):
        """Save the processor to a directory."""
        self.processor.save_pretrained(save_directory, **kwargs)

    def batch_decode(self, *args, **kwargs):
        """Decode token IDs to text."""
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """Decode token IDs to text."""
        return self.tokenizer.decode(*args, **kwargs)
