"""
MTEB wrapper for ColIntern3.5 models.
"""

from __future__ import annotations

import logging
from functools import partial

import torch
from transformers.utils.import_utils import is_flash_attn_2_available

from mteb.model_meta import ModelMeta
from mteb.models.colpali_models import COLPALI_TRAINING_DATA, ColPaliEngineWrapper
from mteb.requires_package import requires_package

logger = logging.getLogger(__name__)


class ColIntern3_5Wrapper(ColPaliEngineWrapper):
    """Wrapper for ColIntern3.5 model."""

    def __init__(
        self,
        model_name: str = "output/checkpoint-1847",  # Default to our trained checkpoint
        revision: str | None = None,
        device: str | None = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ):
        requires_package(
            self, "colpali_engine", model_name, "pip install mteb[colpali_engine]"
        )
        from colpali_engine.models import ColIntern3_5, ColIntern3_5Processor
        from peft import PeftModel
        import os

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Validate that this is a trained checkpoint path
        if not os.path.exists(model_name):
            raise ValueError(f"Model path does not exist: {model_name}")
        
        # Check for required PEFT files
        adapter_config_path = os.path.join(model_name, "adapter_config.json")
        adapter_model_path = os.path.join(model_name, "adapter_model.safetensors")
        
        if not os.path.exists(adapter_config_path):
            raise ValueError(f"adapter_config.json not found in {model_name}. This must be a trained PEFT checkpoint.")
        if not os.path.exists(adapter_model_path):
            raise ValueError(f"adapter_model.safetensors not found in {model_name}. This must be a trained PEFT checkpoint.")
        
        logger.info(f"Loading trained ColIntern3.5 checkpoint from: {model_name}")
        
        # Always load as PEFT checkpoint - no fallbacks to base model
        base_model_name = "OpenGVLab/InternVL3_5-1B-HF"
        
        # Load the base model
        logger.info(f"Loading base model: {base_model_name}")
        self.mdl = ColIntern3_5.from_pretrained(
            base_model_name,
            device_map=self.device,
            torch_dtype=torch_dtype,
            mask_non_image_embeddings=True,
            **kwargs,
        )
        
        # Load the trained PEFT adapter
        logger.info(f"Loading trained adapter from: {model_name}")
        self.mdl = PeftModel.from_pretrained(
            self.mdl, 
            model_name,
            torch_dtype=torch_dtype,
            is_trainable=False
        )
        
        # Merge the adapter weights into the base model
        logger.info("Merging trained adapter weights...")
        self.mdl = self.mdl.merge_and_unload()
        
        self.mdl.eval()
        
        # Validate that we have the trained model by checking custom_text_proj
        if hasattr(self.mdl, 'custom_text_proj'):
            logger.info("✅ Successfully loaded trained model with custom_text_proj")
        else:
            raise RuntimeError("Failed to load trained model - custom_text_proj not found!")
        
        # Load processor from base model
        logger.info(f"Loading processor from base model: {base_model_name}")
        self.processor = ColIntern3_5Processor.from_pretrained(
            base_model_name,
            max_num_visual_tokens=768
        )
        
        # Initialize processor kwargs (inherited from ColPaliEngineWrapper)
        self.processor_kwargs = {}
        
        logger.info(f"Model loaded successfully on device: {self.device}")

    def encode(self, sentences, **kwargs):
        return self.get_text_embeddings(texts=sentences, **kwargs)

    def encode_input(self, inputs):
        return self.mdl(**inputs)


# Training data for ColIntern3.5 (similar to ColPali training data)
COLINTERN3_5_TRAINING_DATA = {
    # Based on the same training set as ColPali models
    "DocVQA": ["train"],
    "InfoVQA": ["train"], 
    "TATDQA": ["train"],
    "arXivQA": ["train"],
}

# Model metadata for the trained ColIntern3.5 model
colintern3_5_1b_lora = ModelMeta(
    loader=partial(
        ColIntern3_5Wrapper,
        model_name="output/checkpoint-1847",                                                    
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
        if is_flash_attn_2_available()
        else None,
    ),
    name="local/colintern3_5-1B-lora",
    languages=["eng-Latn"],                                                  
    revision="checkpoint-1847",  # Our local checkpoint
    release_date="2025-09-05",  # Current date
    n_parameters=905_000_000,  # 905M parameters as shown in training                         
    memory_usage_mb=1800,  # Approximate memory usage in MB
    max_tokens=32768,  # InternVL3.5 context length
    embed_dim=3584,  # Based on InternVL3.5-1B hidden size
    license="apache-2.0",  # Must be lowercase
    open_weights=True,
    public_training_code="https://github.com/illuin-tech/colpali",
    public_training_data="https://huggingface.co/datasets/vidore/colpali_train_set",
    framework=["ColPali"],  # Only include valid frameworks
    reference="https://huggingface.co/OpenGVLab/InternVL3_5-1B-HF",
    similarity_fn_name="max_sim",
    use_instructions=True,
    training_datasets=COLINTERN3_5_TRAINING_DATA,
)
