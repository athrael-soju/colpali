# run_eval.py
from __future__ import annotations

import os
import pathlib
import traceback
import multiprocessing as mp

# ---------- set spawn early ----------
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass  # already set

# ---------- environment knobs (before importing torch/mteb) ----------
os.environ.setdefault("HF_DATASETS_DISABLE_MULTIPROCESSING", "1")
os.environ.setdefault("HF_DATASETS_DISABLE_PYARROW_MEMORY_MAP", "1")
os.environ.setdefault("HF_DATASETS_IN_MEMORY_MAX_SIZE", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Use a disk tmp to avoid /dev/shm
os.environ.setdefault("TMPDIR", "/mnt/c/tmp")
pathlib.Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)

import torch  # noqa: E402

# ---------- CRITICAL: avoid /dev/shm ----------
import torch.multiprocessing as torch_mp  # noqa: E402
try:
    torch_mp.set_sharing_strategy("file_descriptor")  # no torch_shm_manager
except Exception:
    pass

# Make DataLoader never use shared memory fastpath
import torch.utils.data.dataloader as _dl  # noqa: E402
try:
    _dl._use_shared_memory = False  # type: ignore[attr-defined]
except Exception:
    pass

# Monkey-patch DataLoader to force single-process everywhere
from torch.utils.data import DataLoader as _OrigDL  # noqa: E402
class _SingleProcessDL(_OrigDL):
    def __init__(self, *args, **kwargs):
        kwargs["num_workers"] = 0
        kwargs["persistent_workers"] = False
        kwargs["pin_memory"] = False
        super().__init__(*args, **kwargs)

import torch.utils.data  # noqa: E402
torch.utils.data.DataLoader = _SingleProcessDL  # type: ignore[assignment]

torch.set_num_threads(1)

import mteb  # noqa: E402
from datetime import date  # noqa: E402
from functools import partial  # noqa: E402
from mteb.model_meta import ModelMeta  # noqa: E402
from colintern_models import ColInternWrapper  # noqa: E402


def print_gpu_info(prefix: str = ""):
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name()
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        alloc = torch.cuda.memory_allocated(0) / 1e9
        cached = torch.cuda.memory_reserved(0) / 1e9
        print(f"{prefix}CUDA available: {name}")
        print(f"{prefix}CUDA memory: {total_gb:.1f} GB")
        print(f"{prefix}CUDA memory allocated: {alloc:.1f} GB")
        print(f"{prefix}CUDA memory cached: {cached:.1f} GB")
    else:
        print(f"{prefix}CUDA not available, using CPU")


def tasks_from_benchmark(bench):
    if hasattr(bench, "tasks") and bench.tasks:
        return bench.tasks
    if hasattr(bench, "create_tasks"):
        return bench.create_tasks()
    return [bench]


def main():
    # === Configuration ===
    DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
    MODEL_NAME = "./outputs/colintern3_5-run1"
    BENCHMARKS = ["ViDoRe(v1)", "ViDoRe(v2)"]

    MAX_VIS_TOKENS = int(os.getenv("MAX_VIS_TOKENS", "768"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))
    NUM_WORKERS = 0  # hard zero (we also monkey-patch globally)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print_gpu_info()

    custom_model_meta = ModelMeta(
        loader=partial(
            ColInternWrapper,
            model_name=MODEL_NAME,
            revision=None,
            torch_dtype=DTYPE,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            pin_memory=False,
            max_num_visual_tokens=MAX_VIS_TOKENS,
        ),
        name=f"local/colintern1b:{MODEL_NAME.split('/')[-1]}",
        languages=["eng-Latn"],
        revision="local",
        release_date=date(2025, 9, 4),
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
    custom_model = custom_model_meta.load_model()

    # Reinforce single-process knobs
    custom_model.batch_size = BATCH_SIZE
    custom_model.num_workers = NUM_WORKERS
    custom_model.pin_memory = False

    device = getattr(custom_model, "device", None)
    if device is None:
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    custom_model.processor_kwargs = {"device": device}

    print("Model loaded successfully!")
    print("mteb version:", getattr(mteb, "__version__", "unknown"))

    tasks = mteb.get_benchmarks(names=BENCHMARKS)
    evaluator = mteb.MTEB(tasks=tasks)
    results = evaluator.run(custom_model)
    print(results)


if __name__ == "__main__":
    main()
