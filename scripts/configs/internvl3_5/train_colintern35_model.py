
# train_colintern35_model.py
# ColPali-style training for InternVL3.5-1B using colpali_engine (datasets, losses, trainers).

# Scouting example (≈2–3h on 1×5090, ~40s/step):
# python scripts/configs/internvl3_5/train_colintern35_model.py \
#   --stage finetune \
#   --backbone OpenGVLab/InternVL3_5-1B-HF \
#   --output-dir ./runs/internvl35_scout \
#   --peft --loss ce \
#   --epochs 1 --max-steps 200 \
#   --lr 1e-5 --warmup-steps 50 \
#   --per-device-train-batch-size 16 \
#   --gradient-accumulation-steps 8 \
#   --eval-steps 50 --save-steps 200 --logging-steps 5 \
#   --save-total-limit 2 \
#   --max-image-size 448
#
# Full run (after a good scout):
# python scripts/configs/internvl3_5/train_colintern35_model.py \
#   --stage finetune \
#   --backbone OpenGVLab/InternVL3_5-1B-HF \
#   --output-dir ./runs/internvl35_go \
#   --peft --loss ce \
#   --epochs 1 --max-steps 1000 \
#   --lr 1e-5 --warmup-steps 100 \
#   --per-device-train-batch-size 16 \
#   --gradient-accumulation-steps 8 \
#   --eval-steps 100 --save-steps 300 --logging-steps 10 \
#   --save-total-limit 3 \
#   --max-image-size 448

import argparse
import sys
import re
from pathlib import Path
import multiprocessing as mp

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import TrainingArguments

# Ensure 'colpali/' directory is on sys.path so 'colpali_engine' is importable when running this file
_THIS_FILE = Path(__file__).resolve()
_COLPALI_DIR = _THIS_FILE.parents[3]  # .../colpali
if str(_COLPALI_DIR) not in sys.path:
    sys.path.insert(0, str(_COLPALI_DIR))

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ---------- CRITICAL: avoid /dev/shm ----------
try:
    mp.set_sharing_strategy("file_descriptor")  # no torch_shm_manager
except Exception:
    pass

from colpali_engine.data.dataset import ColPaliEngineDataset
from colpali_engine.loss.late_interaction_losses import ColbertLoss, ColbertPairwiseCELoss
from colpali_engine.models import ColIntern3_5, ColIntern3_5Processor
from colpali_engine.trainer.colmodel_torch_training import ColModelTorchTraining
from colpali_engine.trainer.colmodel_training import ColModelTraining, ColModelTrainingConfig
from colpali_engine.utils.dataset_transformation import load_train_set


def get_target_modules_by_regex(model):
    """
    Generate target modules for LoRA using regex pattern (Qwen-like blocks).
    Pattern matches:
    - language_model layers (excluding visual): down_proj, gate_proj, up_proj, k_proj, q_proj, v_proj, o_proj
    - custom_text_proj
    """
    pattern = r"(.*(language_model)(?!.*visual).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$|.*(custom_text_proj).*$)"
    all_modules = dict(model.named_modules())
    target_modules = []
    for name, module in all_modules.items():
        if hasattr(module, 'weight') and re.match(pattern, name):
            target_modules.append(name)
    print(f"[LoRA] Regex matched {len(target_modules)} target modules.")
    if target_modules:
        print("       Sample:", target_modules[:5])
    return target_modules


def parse_args():
    p = argparse.ArgumentParser()
    # Stage/backbone
    p.add_argument("--stage", type=str, default="base", choices=["base", "finetune"], help="training stage")
    p.add_argument("--backbone", type=str, default="OpenGVLab/InternVL3_5-1B-HF", help="HF model id")
    p.add_argument("--output-dir", type=str, default="./runs/internvl35", help="where to write model + script copy")
    # Loss & PEFT
    p.add_argument("--peft", action="store_true", help="use PEFT (LoRA)")
    p.add_argument("--loss", type=str, default="ce", choices=["ce", "pairwise"], help="loss function to use")
    # HP & runtime
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=200, help="cap total training steps (use -1 to disable)")
    p.add_argument("--lr", type=float, default=5e-5, help="learning rate")
    p.add_argument("--warmup-ratio", type=float, default=0.025)
    p.add_argument("--warmup-steps", type=int, default=50, help="override warmup by fixed steps; set 0 to use ratio only")
    p.add_argument("--per-device-train-batch-size", type=int, default=16)
    p.add_argument("--per-device-eval-batch-size", type=int, default=16)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--eval-steps", type=int, default=50)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--logging-steps", type=int, default=5)
    p.add_argument("--save-total-limit", type=int, default=3)
    p.add_argument("--report-to", type=str, default="none", help="wandb|tensorboard|none")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-image-size", type=int, default=448, help="processor image size (long edge)")
    # ColBERT loss temperature (CE)
    p.add_argument("--tau", type=float, default=0.02, help="temperature for CE loss")
    # Trainer backend
    p.add_argument("--trainer", type=str, default="hf", choices=["torch", "hf"], help="trainer to use")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)

    # ---- Loss ----
    if args.loss == "ce":
        loss_func = ColbertLoss(
            temperature=args.tau,
            normalize_scores=True,
            use_smooth_max=False,
            pos_aware_negative_filtering=False,
        )
    elif args.loss == "pairwise":
        loss_func = ColbertPairwiseCELoss(
            normalize_scores=False,
        )
    else:
        raise ValueError(f"Unknown loss function: {args.loss}")

    # ---- Model ----
    base_model = ColIntern3_5.from_pretrained(
        pretrained_model_name_or_path=args.backbone,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
        device_map="auto",  # ensure FA2 kernels bind on-GPU immediately
    )
    # Silence use_cache with gradient checkpointing
    if hasattr(base_model, "config"):
        try:
            base_model.config.use_cache = False
        except Exception:
            pass

    # ---- LoRA targets ----
    target_modules = get_target_modules_by_regex(base_model) if args.peft else None

    # ---- Datasets ----
    train_dataset = load_train_set()
    eval_dataset = ColPaliEngineDataset(
        load_dataset("./data_dir/colpali_train_set", split="test"), pos_target_column_name="image"
    )

    # ---- Processor ----
    processor = ColIntern3_5Processor.from_pretrained(
        pretrained_model_name_or_path=args.backbone,
        max_image_size=args.max_image_size,
        downsample_ratio=getattr(base_model.config, "downsample_ratio", 0.5),  # keep processor/model aligned
    )

    # ---- Stage-aware defaults ----
    if args.stage == "finetune":
        if args.lr == 5e-5:
            args.lr = 1e-5
        if args.gradient_accumulation_steps == 4:
            args.gradient_accumulation_steps = 8

    # Normalize max_steps for HF (use -1 to disable cap)
    max_steps = args.max_steps if args.max_steps is not None else -1

    tr_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        max_steps=max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        eval_strategy="steps",
        dataloader_num_workers=8,
        dataloader_prefetch_factor=4,
        dataloader_persistent_workers=True,
        dataloader_pin_memory=True,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        warmup_ratio=(0.0 if (args.warmup_steps and args.warmup_steps > 0) else args.warmup_ratio),
        warmup_steps=(args.warmup_steps if (args.warmup_steps and args.warmup_steps > 0) else 0),
        learning_rate=args.lr,
        lr_scheduler_type="linear",
        save_total_limit=args.save_total_limit,
        bf16=True,
        optim="adamw_bnb_8bit",
        tf32=True,
        report_to=None if args.report_to == "none" else args.report_to,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    # ---- LoRA config ----
    peft_config = None
    if args.peft:
        peft_config = LoraConfig(
            r=32,
            lora_alpha=32,
            lora_dropout=0.05 if args.stage == "base" else 0.1,
            init_lora_weights="gaussian",
            bias="none",
            task_type="FEATURE_EXTRACTION",
            target_modules=target_modules,
            modules_to_save=["custom_text_proj"],  # ensure projection is saved
        )
        print(f"[LoRA] using r=32, alpha=32, dropout={0.05 if args.stage=='base' else 0.1}")

    # ---- Trainer wiring ----
    config = ColModelTrainingConfig(
        output_dir=args.output_dir,
        processor=processor,
        model=base_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        run_eval=True,
        loss_func=loss_func,
        tr_args=tr_args,
        peft_config=peft_config,
    )

    trainer = ColModelTraining(config) if args.trainer == "hf" else ColModelTorchTraining(config)
    trainer.train()
    trainer.save()
