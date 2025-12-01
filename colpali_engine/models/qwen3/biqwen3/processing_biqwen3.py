from typing import List, Optional, Union

import torch
from transformers import BatchEncoding, BatchFeature

from colpali_engine.models.qwen3.colqwen3 import ColQwen3Processor


class BiQwen3Processor(ColQwen3Processor):
    """
    Processor for BiQwen3.

    This is a thin wrapper around ColQwen3Processor that overrides the scoring method
    to use single-vector similarity (cosine similarity) instead of multi-vector MaxSim.
    """

    def process_texts(self, texts: List[str]) -> Union[BatchFeature, BatchEncoding]:
        """
        Process texts for BiQwen3.

        Args:
            texts: List of input texts.

        Returns:
            Union[BatchFeature, BatchEncoding]: Processed texts.
        """
        return self.processor(
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
        Compute the cosine similarity for the given query and passage embeddings.

        Unlike ColQwen3Processor which uses MaxSim for multi-vector scoring,
        BiQwen3Processor uses simple cosine similarity for single-vector scoring.

        Args:
            qs: List of query embeddings, each of shape (hidden_size,) or (1, hidden_size).
            ps: List of passage embeddings, each of shape (hidden_size,) or (1, hidden_size).
            device: Device to perform computation on.
            **kwargs: Additional arguments (ignored).

        Returns:
            torch.Tensor: Similarity scores of shape (num_queries, num_passages).
        """
        return self.score_single_vector(qs, ps, device=device)
