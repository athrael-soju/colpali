# coding=utf-8
# Processor for ColIntern3.5 (InternVL3.5-based retriever)

from __future__ import annotations

from typing import ClassVar, List, Optional, Tuple, Union

import torch
from PIL import Image

from transformers.models.internvl.processing_internvl import InternVLProcessor  # type: ignore
from transformers import BatchEncoding, BatchFeature
from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize


class ColIntern3_5_Processor(BaseVisualRetrieverProcessor, InternVLProcessor):  # noqa: N801
    """
    Processor tailored for ColIntern3.5 training & inference.

    It exposes convenience helpers:
      * process_images(images)  -> returns {'pixel_values': tensor}
      * process_texts(texts)    -> returns {'input_ids', 'attention_mask'}
      * score_multi_vector(qs, ps) for ColBERT-style late interaction (MaxSim) scoring
    """

    # Not used for encoding, but kept for API symmetry with ColQwen processors
    visual_prompt_prefix: ClassVar[str] = ""
    # Provide defaults to match BaseVisualRetrieverProcessor expectations.
    query_prefix: ClassVar[str] = "Query: "

    @property
    def query_augmentation_token(self) -> str:
        """
        Token used to pad/augment queries for reasoning buffer (similar to ColPali).
        Defaults to tokenizer.pad_token when available.
        """
        tok = getattr(self, "tokenizer", None)
        return getattr(tok, "pad_token", "") if tok is not None else ""

    @classmethod
    def from_pretrained(
        cls,
        *args,
        device_map: Optional[str] = None,
        **kwargs,
    ):
        instance = super().from_pretrained(
            *args,
            device_map=device_map,
            **kwargs,
        )

        if "max_num_visual_tokens" in kwargs:
            instance.image_processor.max_pixels = kwargs["max_num_visual_tokens"] * 28 * 28
            instance.image_processor.size["longest_edge"] = instance.image_processor.max_pixels

        return instance


    def process_images(self, images: List[Image.Image]) -> Union[BatchFeature, BatchEncoding]:
        images = [im.convert("RGB") for im in images]
        image_inputs = self.image_processor(images=images, return_tensors="pt")
        # Return only what's needed
        return BatchFeature(data={"pixel_values": image_inputs.get("pixel_values")})

    def process_texts(self, texts: List[str], max_length: Optional[int] = None) -> Union[BatchFeature, BatchEncoding]:
        enc = self.tokenizer(
            texts,
            padding="longest",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return BatchEncoding(enc.data)

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
        spatial_merge_size: int,
    ) -> Tuple[int, int]:
        """
        Get the number of patches (n_patches_x, n_patches_y) that will be used to process an image of
        size (height, width) with the given patch size.

        The `spatial_merge_size` is the number of patches that will be merged spatially. It is stored in
        as a `Qwen2VLForConditionalGeneration` attribute under `model.spatial_merge_size`.
        """
        patch_size = self.image_processor.patch_size

        height_new, width_new = smart_resize(
            width=image_size[0],
            height=image_size[1],
            factor=patch_size * self.image_processor.merge_size,
            min_pixels=self.image_processor.size["shortest_edge"],
            max_pixels=self.image_processor.size["longest_edge"],
        )

        n_patches_x = width_new // patch_size // spatial_merge_size
        n_patches_y = height_new // patch_size // spatial_merge_size

        return n_patches_x, n_patches_y

    def get_image_mask(self, batch_images: BatchFeature) -> torch.Tensor:
        return batch_images.input_ids == self.image_token_id