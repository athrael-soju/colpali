# coding=utf-8
# Processor for ColIntern3.5 (InternVL3.5-based retriever)

from __future__ import annotations

from typing import ClassVar, List, Optional, Tuple, Union

import torch
from PIL import Image

from transformers.models.internvl import InternVLProcessor  # type: ignore
from transformers import BatchEncoding, BatchFeature

from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor


class ColIntern3_5_Processor(BaseVisualRetrieverProcessor, InternVLProcessor):  # noqa: N801
    """
    Processor tailored for ColIntern3.5 training & inference.

    It exposes convenience helpers:
      * process_images(images)  -> returns {'pixel_values': tensor}
      * process_texts(texts)    -> returns {'input_ids', 'attention_mask'}
      * score_multi_vector(qs, ps) for ColBERT-style late interaction (MaxSim) scoring

    Notes
    -----
    • This implementation standardizes the image processor `size` to the fixed-size schema:
        size = {"height": 448, "width": 448}
      This avoids the ValueError raised when mixing {'height','width'} with {'shortest_edge','longest_edge'}.

    • If you want adaptive shortest/longest-edge later, wire it in explicitly and remove
      the fixed {height,width} below (don’t mix schemas).
    """

    visual_prompt_prefix: ClassVar[str] = "<image><bos>Describe the image."
    query_prefix: ClassVar[str] = "Query: "

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def query_augmentation_token(self) -> str:
        """
        Return the query augmentation token.
        Query augmentation buffers are used as reasoning buffers during inference.
        """
        return self.tokenizer.pad_token

    # -------------------------
    # Construction / Sanitizing
    # -------------------------
    @classmethod
    def from_pretrained(
        cls,
        *args,
        device_map: Optional[str] = None,
        **kwargs,
    ):
        """
        Load the base InternVL processor and sanitize its image size to a valid fixed-size schema.
        If `max_num_visual_tokens` is provided, only `max_pixels` (area cap) is set; we DO NOT
        inject 'longest_edge' into `size` to avoid invalid combinations.
        """
        instance = super().from_pretrained(*args, device_map=device_map, **kwargs)

        # Carry an area budget if the caller passes it (no schema mix).
        if "max_num_visual_tokens" in kwargs:
            # InternVL uses ~28x28 per 14-px patch with merge_size=2; maintain your convention.
            instance.image_processor.max_pixels = int(kwargs["max_num_visual_tokens"]) * 28 * 28

        # ---- IMPORTANT: sanitize size schema to fixed {height, width} only ----
        ip = instance.image_processor
        # Default InternVL HF ports: 448 x 448
        default_h, default_w = 448, 448

        # Ensure dict and drop any unsupported keys (e.g., 'shortest_edge', 'longest_edge', etc.)
        size_dict = {}
        if isinstance(getattr(ip, "size", None), dict):
            # Keep only height/width if present; otherwise inject sensible defaults.
            h = int(ip.size.get("height", default_h))
            w = int(ip.size.get("width", default_w))
            size_dict = {"height": h, "width": w}
        else:
            size_dict = {"height": default_h, "width": default_w}

        ip.size = size_dict  # now guaranteed valid for the fixed-size path

        return instance

    # ---------------
    # Public Helpers
    # ---------------
    def process_images(self, images: List[Image.Image]) -> Union[BatchFeature, BatchEncoding]:
        """
        Convert a list of PIL images into a batch of pixel_values suitable for the model.
        """
        images = [im.convert("RGB") for im in images]
        image_inputs = self.image_processor(images=images, return_tensors="pt")
        # Return only what's needed by your model
        return BatchFeature(data={"pixel_values": image_inputs.get("pixel_values")})

    def process_texts(self, texts: List[str], max_length: Optional[int] = None) -> Union[BatchFeature, BatchEncoding]:
        """
        Tokenize a list of texts into input_ids and attention_mask.
        """
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

    # -------------------------
    # Patch layout (fixed-size)
    # -------------------------
    def get_n_patches(
        self,
        image_size: Tuple[int, int],
        spatial_merge_size: int,
    ) -> Tuple[int, int]:
        """
        Compute the number of patches (n_patches_x, n_patches_y) for the *resized* image that the
        image processor will actually feed to the vision encoder, assuming a fixed {height,width} schema.

        Parameters
        ----------
        image_size : (H, W)
            The original image size; we ignore it here because the processor *always* resizes to
            ip.size['height'] x ip.size['width'] in this fixed-size schema.
        spatial_merge_size : int
            Vision grid spatial merge factor (e.g., 2 for InternVL).

        Returns
        -------
        (n_patches_x, n_patches_y)
            Number of patch vectors along width (x) and height (y) after resizing & merging.
        """
        ip = self.image_processor

        # 1) Determine the processed spatial size (the fixed target size)
        target_h = int(ip.size.get("height", 448))
        target_w = int(ip.size.get("width", 448))

        # 2) Patch/merge sizes (fall back to common InternVL defaults if missing)
        patch_size = int(getattr(ip, "patch_size", 14))
        merge_size = int(getattr(ip, "merge_size", 2))

        # 3) Effective “stride” in pixels the model uses per embedding vector
        stride = patch_size * int(spatial_merge_size)

        # 4) Derived grid sizes (how many embeddings along each axis)
        n_patches_x = target_w // stride
        n_patches_y = target_h // stride

        return n_patches_x, n_patches_y

    # Kept for API parity; not used by the image branch but harmless
    def get_image_mask(self, batch_images: BatchFeature) -> torch.Tensor:
        return batch_images.input_ids == self.image_token_id
