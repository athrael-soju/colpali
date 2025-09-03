# coding=utf-8
# ColIntern3.5: InternVL3.5-based multi-vector retriever (ColBERT-style)
# Copyright 2025
# Licensed under the Apache 2.0 License.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import nn

from transformers.models.internvl.modeling_internvl import (  # type: ignore
    InternVLModel,
    InternVLConfig,
    InternVLVisionModel,
    InternVLPreTrainedModel,
)

from transformers.utils import ModelOutput


@dataclass
class ColInternOutput(ModelOutput):
    """Output of ColIntern3.5 forward pass.

    Args:
        loss: Optional training loss (when both query and document are provided).
        query_embeddings: Query token embeddings of shape (batch, M_tokens, 128) if computed.
        doc_embeddings: Document patch embeddings of shape (batch, N_patches, 128) if computed.
    """
    loss: Optional[torch.FloatTensor] = None
    query_embeddings: Optional[torch.FloatTensor] = None
    doc_embeddings: Optional[torch.FloatTensor] = None


class ColIntern3_5(InternVLModel):  # noqa: N801
    """ColIntern 3.5 model.

    This class wraps the InternVL3.5 backbone and adds:
      - 128-dim projection heads for vision patches and text tokens
      - ColBERT-style late interaction utilities
      - A training-mode forward that can compute the late-interaction CE loss with in-batch negatives

    Notes
    -----
    * Vision encoding uses the InternVL *vision_tower* (ViT) directly to obtain patch-level features.
      We exclude the [CLS] token and project each patch to 128-d.
    * Text encoding uses the InternVL *language_model* to obtain token hidden states,
      then projects to 128-d.

    * All projections are L2-normalized so dot product equals cosine similarity.

    * FlashAttention-2 is enabled by passing `attn_implementation="flash_attention_2"` to
      `.from_pretrained(...)`.
    """

    # allow loading checkpoints saved under InternVLForConditionalGeneration (keys prefixed by 'model.')
    _checkpoint_conversion_mapping = {
        r"^model\.vision_tower": "vision_tower",
        r"^model\.multi_modal_projector": "multi_modal_projector",
        r"^model\.language_model": "language_model",
    }

    main_input_name = "doc_pixel_values"  # used by HF Trainer when batching documents

    def __init__(self, config: InternVLConfig, output_dim: int = 128, mask_non_image_embeddings: bool = False):
        super().__init__(config)
        self.output_dim = int(output_dim)
        self.mask_non_image_embeddings = bool(mask_non_image_embeddings)

        # Separate projection heads (vision/text may have different hidden sizes)
        self.vision_proj = nn.Linear(config.vision_config.hidden_size, self.output_dim, bias=False)
        self.text_proj = nn.Linear(config.text_config.hidden_size, self.output_dim, bias=False)

        # initialize newly added weights
        self.post_init()

    # --------------------------
    # Encoding helpers
    # --------------------------
    @torch.no_grad()
    def embed_documents(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode documents (images) into multi-vector patch embeddings (B, N_patches, 128).

        Args:
            pixel_values: float tensor (B, 3, H, W)

        Returns:
            doc_embeddings: (B, N_patches, 128) L2-normalized
        """
        self.eval()
        return self._encode_images(pixel_values)

    @torch.no_grad()
    def embed_queries(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode text queries into multi-vector token embeddings (B, M_tokens, 128)."""
        self.eval()
        return self._encode_queries(input_ids, attention_mask)

    def _encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # Obtain patch-level embeddings from InternVL vision tower
        # Returned shape: (B, 1+N_patches, H_dim). We drop the first [CLS]-like token.
        vision_out = self.vision_tower(pixel_values=pixel_values)
        patch_states = vision_out.last_hidden_state[:, 1:, :]  # (B, N_patches, H_dim)

        # Project to 128-d and L2-normalize
        proj = self.vision_proj(patch_states)  # (B, N_patches, 128)
        proj = proj / (proj.norm(dim=-1, keepdim=True) + 1e-12)
        return proj

    def _encode_queries(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Text encoder forward (no images); obtain last hidden states per token
        # Disable KV cache & request hidden states for stable training encodings.
        outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state  # (B, M, H_dim)

        # Project to 128-d and L2-normalize; mask pads
        proj = self.text_proj(hidden)  # (B, M, 128)
        proj = proj / (proj.norm(dim=-1, keepdim=True) + 1e-12)
        if attention_mask is not None:
            proj = proj * attention_mask.unsqueeze(-1)
        return proj

    @staticmethod
    def _late_interaction_scores(q: torch.Tensor, d: torch.Tensor, q_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute pairwise late-interaction scores S in R^{B_q x B_d}.

        Args:
            q: (Bq, M, 128)  - query token embeddings
            d: (Bd, N, 128)  - doc patch embeddings
            q_mask: (Bq, M)  - 1 for valid tokens, 0 for padding

        Returns:
            scores: (Bq, Bd)
        """
        # sim[q, d, m, n] = dot(q[q, m], d[d, n])
        sim = torch.einsum("qmd,pnd->qmpn", q, d)  # (Bq, Bd, M, N)
        max_sim = sim.max(dim=-1).values               # (Bq, Bd, M)
        if q_mask is not None:
            max_sim = max_sim * q_mask.unsqueeze(1)    # broadcast over docs
        scores = max_sim.sum(dim=-1)                   # (Bq, Bd)
        return scores

    def forward(
        self,
        # Document (image) inputs
        doc_pixel_values: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,  # alias for convenience
        # Query (text) inputs
        query_input_ids: Optional[torch.Tensor] = None,
        query_attention_mask: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,     # alias
        attention_mask: Optional[torch.Tensor] = None, # alias
        # Training options
        compute_loss: bool = False,
        temperature: float = 0.02,
        **kwargs,
    ) -> ColInternOutput:
        """Two modes:
        1) Embedding mode: pass only `doc_pixel_values` OR only `query_input_ids` -> returns embeddings.
        2) Training mode: pass both `doc_pixel_values` and `query_input_ids` -> returns late-interaction CE loss.
        """
        # Map aliases if provided
        if pixel_values is not None and doc_pixel_values is None:
            doc_pixel_values = pixel_values
        if input_ids is not None and query_input_ids is None:
            query_input_ids = input_ids
            query_attention_mask = attention_mask

        has_docs = doc_pixel_values is not None
        has_queries = query_input_ids is not None

        if not has_docs and not has_queries:
            raise ValueError("ColIntern3_5.forward() expects either documents, queries, or both.")

        doc_embeds: Optional[torch.Tensor] = None
        query_embeds: Optional[torch.Tensor] = None

        if has_docs:
            doc_embeds = self._encode_images(doc_pixel_values.to(dtype=self.dtype, device=self.device))

        if has_queries:
            if query_attention_mask is None:
                raise ValueError("query_attention_mask must be provided when encoding queries.")
            query_input_ids = query_input_ids.to(device=self.device)
            query_attention_mask = query_attention_mask.to(device=self.device)
            query_embeds = self._encode_queries(query_input_ids, query_attention_mask)

        # If only one modality is requested, return the raw embeddings tensor
        if has_docs and not has_queries:
            # (B, N_patches, D)
            return doc_embeds  # type: ignore[return-value]
        if has_queries and not has_docs:
            # (B, M_tokens, D)
            return query_embeds  # type: ignore[return-value]

        loss = None
        if has_docs and has_queries:
            # In-batch negatives: compute B x B similarity matrix and CE against diagonal
            scores = self._late_interaction_scores(query_embeds, doc_embeds, q_mask=query_attention_mask)
            if temperature is not None and temperature > 0:
                scores = scores / temperature
            labels = torch.arange(scores.size(0), device=scores.device)
            loss = nn.functional.cross_entropy(scores, labels)

        return ColInternOutput(loss=loss, query_embeddings=query_embeds, doc_embeddings=doc_embeds)

    # Convenience properties (useful for processors / downstream code)
    @property
    def patch_size(self) -> int:
        return int(self.config.vision_config.patch_size[0]) if isinstance(self.config.vision_config.patch_size, (list, tuple)) else int(self.config.vision_config.patch_size)
