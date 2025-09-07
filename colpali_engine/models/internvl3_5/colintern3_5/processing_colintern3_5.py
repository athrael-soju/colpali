from typing import ClassVar, List, Optional, Tuple, Union

import torch
from PIL import Image
from transformers import BatchEncoding, BatchFeature, InternVLProcessor

from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor


class ColIntern3_5Processor(BaseVisualRetrieverProcessor, InternVLProcessor):
    """
    Processor for ColIntern3_5.
    """

    visual_prompt_prefix: ClassVar[str] = "<IMG_CONTEXT>\nDescribe the image."

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def query_augmentation_token(self) -> str:
        """
        Return the query augmentation token.
        Query augmentation buffers are used as reasoning buffers during inference.
        """
        return self.tokenizer.pad_token

    def process_images(
        self,
        images: List[Image.Image],
    ) -> Union[BatchFeature, BatchEncoding]:
        """
        Process images for ColIntern3_5.

        Args:
            images: List of PIL images.
        """
        images = [image.convert("RGB") for image in images]

        # For each image, create a text prompt with exactly one <image> token
        texts = [self.visual_prompt_prefix for _ in images]
        
        batch_doc = self(
            text=texts,
            images=images,
            return_tensors="pt",
            padding="longest",
        )
        return batch_doc

    def process_texts(self, texts: List[str]) -> Union[BatchFeature, BatchEncoding]:
        """
        Process texts for ColIntern3_5.

        Args:
            texts: List of input texts.

        Returns:
            Union[BatchFeature, BatchEncoding]: Processed texts.
        """
        return self.tokenizer(
            texts,  # InternVL doesn't need BOS token prefix like PaliGemma
            text_pair=None,
            return_token_type_ids=False,
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
        # InternVL uses fixed image sequence length
        image_seq_length = getattr(self, 'image_seq_length', 256)
        # Approximate patch grid from sequence length
        n_patches_per_side = int(image_seq_length ** 0.5)
        return n_patches_per_side, n_patches_per_side

    def get_image_mask(self, batch_images: BatchFeature) -> torch.Tensor:
        return batch_images.input_ids == self.image_token_id
