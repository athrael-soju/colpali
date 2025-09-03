import os
import torch
import tempfile
import pathlib
import traceback
from functools import partial

# Set multiprocessing method BEFORE importing anything else
import multiprocessing as mp
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

import mteb
from mteb.model_meta import ModelMeta
from colintern_models import ColInternWrapper

# Set environment variables for memory management
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# === Configuration ===
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
MODEL_NAME = "./outputs/colintern3_5-run1"
BENCHMARKS = ["ViDoRe(v1)", "ViDoRe(v2)"]

# Clear CUDA cache at start
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"CUDA available: {torch.cuda.get_device_name()}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.1f} GB")
    print(f"CUDA memory cached: {torch.cuda.memory_reserved(0) / 1e9:.1f} GB")
else:
    print("CUDA not available, using CPU")

def print_memory_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1e9
        cached = torch.cuda.memory_reserved(0) / 1e9
        print(f"GPU Memory - Allocated: {allocated:.1f}GB, Cached: {cached:.1f}GB")

custom_model_meta = ModelMeta(
    loader=partial(
        ColInternWrapper,
        model_name=MODEL_NAME,
        revision=None,
        torch_dtype=DTYPE,
        batch_size=1,  # Reduced batch size
        num_workers=0,  # No multiprocessing
        pin_memory=False,  # Disable pin_memory
        max_num_visual_tokens=768,  # Reduced from 768
    ),
    name=f"local/colintern3.5:{MODEL_NAME.split('/')[-1]}",
    languages=["eng-Latn"],
    revision=None,
    release_date="2025-09-03",
    modalities=["image", "text"],
    n_parameters=1_000_000_000,
    memory_usage_mb=4700,
    max_tokens=32768,
    embed_dim=128,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/OpenGVLab/InternVL",
    public_training_data="https://huggingface.co/datasets/OpenGVLab/MMPR-v1.2",
    framework=["ColPali"],
    reference="https://huggingface.co/OpenGVLab/InternVL3_5-1B-HF",
    similarity_fn_name="max_sim",
    use_instructions=True,
    training_datasets={
        "DocVQA": ["train"],
        "InfoVQA": ["train"],
        "TATDQA": ["train"],
        "arXivQA": ["train"],
    },
)

print("Loading model...")
print_memory_usage()
custom_model = custom_model_meta.load_model()
print_memory_usage()

# Ensure processor_kwargs is set correctly
if hasattr(custom_model, "device"):
    device = custom_model.device
elif torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

custom_model.processor_kwargs = {"device": device}

print("Model loaded successfully!")
print("mteb version:", getattr(mteb, "__version__", "unknown"))

tasks = mteb.get_benchmarks(names=BENCHMARKS)
evaluator = mteb.MTEB(tasks=tasks)
results = evaluator.run(custom_model)
print(results)
