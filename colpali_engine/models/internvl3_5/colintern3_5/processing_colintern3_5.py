# coding=utf-8
# Processor for ColIntern3.5 (InternVL3.5-based retriever)

from __future__ import annotations

from typing import ClassVar, List, Optional, Tuple, Union

import torch
from PIL import Image

try:
    from processing_internvl import InternVLProcessor
except Exception:
    from transformers.models.internvl.processing_internvl import InternVLProcessor  # type: ignore

from transformers import BatchEncoding, BatchFeature


class ColIntern3_5_Processor(InternVLProcessor):  # noqa: N801
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

    # -------- Retrieval scoring (late interaction) --------
    @staticmethod
    def score_multi_vector(
        qs: torch.Tensor,  # (Bq, M, D)
        ps: torch.Tensor,  # (Bd, N, D)
        q_mask: Optional[torch.Tensor] = None,
        device: Optional[Union[str, torch.device]] = None,
        chunk_docs: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute late-interaction scores S in R^{Bq x Bd}: sum_m max_n dot(q_m, p_n).

        Args:
            qs: query token embeddings (Bq, M, D)
            ps: passage/document patch embeddings (Bd, N, D)
            q_mask: optional mask on query tokens (Bq, M)
            device: compute device
            chunk_docs: if provided, compute over documents in chunks to limit memory

        Returns:
            scores: (Bq, Bd)
        """
        device = device or (qs.device if isinstance(qs, torch.Tensor) else "cpu")
        qs = qs.to(device)
        ps = ps.to(device)
        if q_mask is not None:
            q_mask = q_mask.to(device)

        if chunk_docs is None:
            sim = torch.einsum("qmd,pnd->qmpn", qs, ps)  # (Bq, Bd, M, N)
            max_sim = sim.max(dim=-1).values               # (Bq, Bd, M)
            if q_mask is not None:
                max_sim = max_sim * q_mask.unsqueeze(1)
            scores = max_sim.sum(dim=-1)                   # (Bq, Bd)
            return scores

        # Chunk over documents to reduce peak memory
        scores_list = []
        Bd = ps.size(0)
        for start in range(0, Bd, chunk_docs):
            end = min(Bd, start + chunk_docs)
            sim = torch.einsum("qmd,pnd->qmpn", qs, ps[start:end])
            max_sim = sim.max(dim=-1).values
            if q_mask is not None:
                max_sim = max_sim * q_mask.unsqueeze(1)
            scores = max_sim.sum(dim=-1)  # (Bq, end-start)
            scores_list.append(scores)
        return torch.cat(scores_list, dim=1)
