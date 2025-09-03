import mteb
from mteb.model_meta import ModelMeta
from colintern_models import ColInternWrapper
from functools import partial
import torch

# === Configuration ===
MODEL_NAME = "./outputs/colintern3_5-run1"
BENCHMARKS = ["ViDoRe(v1)", "ViDoRe(v2)"]

COLPALI_TRAINING_DATA = {
    # from https://huggingface.co/datasets/vidore/colpali_train_set
    "DocVQA": ["train"],
    "InfoVQA": ["train"],
    "TATDQA": ["train"],
    "arXivQA": ["train"],
}

# === Model Metadata ===
custom_model_meta = ModelMeta(
    loader=partial(
        ColInternWrapper,
        model_name=MODEL_NAME,   # Local checkpoint or model hub path
        revision=None,                     # (Revision commit hash if available)
        torch_dtype=torch.float16,
    ),
    name=f"local/colintern3.5:{MODEL_NAME.split('/')[-1]}",
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
    training_datasets=COLPALI_TRAINING_DATA,
)

# === Load Model ===
custom_model = custom_model_meta.load_model()

# === Load Tasks ===
tasks = mteb.get_benchmarks(names=BENCHMARKS)
evaluator = mteb.MTEB(tasks=tasks)

# === Run Evaluation ===
results = evaluator.run(custom_model)
print(results)