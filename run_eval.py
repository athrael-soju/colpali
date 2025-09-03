import mteb
from mteb.model_meta import ModelMeta
from functools import partial
import torch
from colintern_models import ColInternWrapper

# === Configuration ===
MODEL_NAME = "./outputs/colintern3_5-run1"
BENCHMARKS = ["ViDoRe(v1)", "ViDoRe(v2)"]

COLPALI_TRAINING_DATA = {
    "DocVQA": ["train"],
    "InfoVQA": ["train"],
    "TATDQA": ["train"],
    "arXivQA": ["train"],
}

# Choose dtype: keep fp16 unless you specifically target bf16 GPUs
DTYPE = torch.float16  # or torch.bfloat16 on A100/H100/MI300 etc.

custom_model_meta = ModelMeta(
    loader=partial(
        ColInternWrapper,
        model_name=MODEL_NAME,
        revision=None,
        torch_dtype=DTYPE
    ),
    name=f"local/colintern3.5:{MODEL_NAME.split('/')[-1]}",
    languages=["eng-Latn"],
    revision=None,
    release_date="2025-09-03",
    modalities=["image", "text"],
    n_parameters=1_000_000_000,   # <-- 1B
    memory_usage_mb=4700,         # realistic end-to-end ballpark for eval
    max_tokens=32768,
    embed_dim=128,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/OpenGVLab/InternVL",
    public_training_data="https://huggingface.co/datasets/OpenGVLab/MMPR-v1.2",
    framework=["ColPali"],
    reference="https://huggingface.co/OpenGVLab/InternVL3_5-1B",  # if you're basing on a 1B variant
    similarity_fn_name="max_sim",
    use_instructions=True,
    training_datasets=COLPALI_TRAINING_DATA,
)

# === Load Model ===
custom_model = custom_model_meta.load_model()

# --- IMPORTANT: fix for MTEB's similarity() forwarding **self.processor_kwargs
custom_model.processor_kwargs = {
    "device": custom_model.device if hasattr(custom_model, "device") else (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
}

# === Run ===
tasks = mteb.get_benchmarks(names=BENCHMARKS)
evaluator = mteb.MTEB(tasks=tasks)
results = evaluator.run(custom_model)
print(results)
