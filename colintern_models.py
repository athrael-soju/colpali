from __future__ import annotations

import logging
from functools import partial
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta
from mteb.requires_package import (
    requires_image_dependencies,
    requires_package,
)

logger = logging.getLogger(__name__)


class ColInternEngineWrapper:
    """Base wrapper for `colintern_engine` models. (Analogous to ColPaliEngineWrapper, for InternVL-based models.)"""

    def __init__(
        self,
        model_name: str,
        model_class: type,
        processor_class: type,
        revision: str | None = None,
        device: str | None = None,
        **kwargs,
    ):
        requires_image_dependencies()
        # requires_package(
        #     self, "colintern_engine", model_name, "pip install mteb[colintern_engine]"
        # )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        self.mdl = model_class.from_pretrained(
            model_name,
            device_map=self.device,
            adapter_kwargs={"revision": revision},
            **kwargs,
        )
        self.mdl.eval()

        # Load processor
        self.processor = processor_class.from_pretrained(model_name)

    def encode(self, sentences, **kwargs):
        return self.get_text_embeddings(texts=sentences, **kwargs)

    def encode_input(self, inputs):
        return self.mdl(**inputs)

    def get_image_embeddings(
        self,
        images,
        batch_size: int = 32,
        **kwargs,
    ):
        import torchvision.transforms.functional as F

        all_embeds = []

        # Allow images to be a DataLoader or a list of images
        if isinstance(images, DataLoader):
            iterator = images
        else:
            iterator = DataLoader(images, batch_size=batch_size)

        with torch.no_grad():
            for batch in tqdm(iterator):
                # Convert batch items to PIL Images if they are tensors
                imgs = [
                    F.to_pil_image(b.to("cpu")) if not isinstance(b, Image.Image) else b
                    for b in batch
                ]
                # Process images through the processor, then move to target device
                inputs = self.processor.process_images(imgs).to(self.device)
                outs = self.encode_input(inputs)
                all_embeds.extend(outs.cpu().to(torch.float32))

        # Pad sequences of embeddings to have equal length
        padded = torch.nn.utils.rnn.pad_sequence(
            all_embeds, batch_first=True, padding_value=0
        )
        return padded

    def get_text_embeddings(
        self,
        texts,
        batch_size: int = 32,
        **kwargs,
    ):
        all_embeds = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size)):
                batch_texts = texts[i : i + batch_size]
                # Prepend query prefix and append augmentation token (repeated 10 times) to each text
                batch = [
                    self.processor.query_prefix
                    + t
                    + self.processor.query_augmentation_token * 10
                    for t in batch_texts
                ]
                inputs = self.processor.process_texts(batch).to(self.device)
                outs = self.encode_input(inputs)
                all_embeds.extend(outs.cpu().to(torch.float32))

        padded = torch.nn.utils.rnn.pad_sequence(
            all_embeds, batch_first=True, padding_value=0
        )
        return padded

    def get_fused_embeddings(
        self,
        texts: list[str] | None = None,
        images: list[Image.Image] | DataLoader | None = None,
        *,
        task_name: str | None = None,
        prompt_type: PromptType | None = None,
        batch_size: int = 32,
        fusion_mode="sum",
        **kwargs: Any,
    ):
        raise NotImplementedError(
            "Fused embeddings are not supported yet. Please use get_text_embeddings or get_image_embeddings."
        )

    def calculate_probs(self, text_embeddings, image_embeddings):
        scores = self.similarity(text_embeddings, image_embeddings).T
        return scores.softmax(dim=-1)

    def similarity(self, a, b):
        # Use the processor's scoring function to compute similarities
        return self.processor.score(a, b, **self.processor_kwargs)


class ColInternWrapper(ColInternEngineWrapper):
    """Wrapper for ColIntern3.5 models (InternVL3.5-based multi-modal model)."""

    def __init__(
        self,
        model_name: str = "vidore/colintern3.5",
        revision: str | None = None,
        device: str | None = None,
        **kwargs,
    ):
        # requires_package(
        #     self, "colintern_engine", model_name, "pip install mteb[colintern_engine]"
        # )
        # Import the model and processor classes for ColIntern3.5
        from colpali_engine.models.internvl3_5.colintern3_5.processing_colintern3_5 import ColIntern3_5_Processor
        from colpali_engine.models.internvl3_5.colintern3_5.modeling_colintern3_5 import ColIntern3_5

        super().__init__(
            model_name=model_name,
            model_class=ColIntern3_5,
            processor_class=ColIntern3_5_Processor,
            revision=revision,
            device=device,
            **kwargs,
        )


# (If known, specify the datasets used in training ColIntern3.5. Otherwise leave empty or update when available.)
COLINTERN_TRAINING_DATA = {
    # ColIntern3.5 was trained on a combination of large-scale text and image datasets (e.g., LAION, web image-text corpora, etc.)
    # as well as multimodal instruction tuning data (e.g., MMPR-v1.2 for preference optimization).
    # Specific training datasets to be filled in when known.
}

colintern_v3_5 = ModelMeta(
    loader=partial(
        ColInternWrapper,
        model_name="vidore/colintern3.5",   # Local checkpoint or model hub path
        revision=None,                     # (Revision commit hash if available)
        torch_dtype=torch.float16,
    ),
    name="vidore/colintern3.5",
    languages=["eng-Latn"],
    revision=None,
    release_date="2025-09-03",
    modalities=["image", "text"],
    n_parameters=8_500_000_000,      # Approximate total parameters (e.g., ~8.5B for InternVL3.5 8B model)
    memory_usage_mb=15000,          # (Estimated memory usage in MB for FP16 model load – to be confirmed)
    max_tokens=32768,               # Max sequence length (context window), InternVL3.5 supports up to 32k tokens
    embed_dim=128,                  # Embedding dimension for each token vector (assumed 128 as in ColPali/ColBERT approach)
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/OpenGVLab/InternVL",
    public_training_data="https://huggingface.co/datasets/OpenGVLab/MMPR-v1.2",
    framework=["ColPali"],
    reference="https://huggingface.co/OpenGVLab/InternVL3_5-8B",
    similarity_fn_name="max_sim",
    use_instructions=True,
    training_datasets=COLINTERN_TRAINING_DATA,
)
