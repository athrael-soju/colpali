#!/usr/bin/env python3
"""
Standalone training script for ColQwen3 model.

This script provides a programmatic way to train ColQwen3 without YAML configuration.
It can be run directly or customized for specific training needs.

Usage:
    python scripts/train/train_colqwen3_model.py --output-dir ./models/colqwen3 --peft

    # With custom base model
    python scripts/train/train_colqwen3_model.py \
        --base-model Qwen/Qwen3-VL-4B \
        --output-dir ./models/colqwen3 \
        --peft

    # Distributed training
    torchrun --nproc_per_node=8 scripts/train/train_colqwen3_model.py \
        --output-dir ./models/colqwen3 \
        --peft
"""

import argparse
import os
from typing import Optional

import torch
from peft import LoraConfig
from transformers import TrainingArguments

from colpali_engine.loss.late_interaction_losses import ColbertPairwiseCELoss
from colpali_engine.models import ColQwen3, ColQwen3Processor
from colpali_engine.trainer.colmodel_training import ColModelTraining, ColModelTrainingConfig
from colpali_engine.utils.dataset_transformation import load_train_set
from colpali_engine.utils.gpu_stats import print_gpu_utilization


def parse_args():
    parser = argparse.ArgumentParser(description="Train ColQwen3 model")
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen3-VL-2B-Instruct",
        help="Base model path or HuggingFace model ID (e.g., Qwen/Qwen3-VL-2B-Instruct or Qwen/Qwen3-VL-8B-Instruct)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models/colqwen3-trained",
        help="Directory to save the trained model",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=128,
        help="Embedding dimension for ColQwen3 (default: 128, TomoroAI uses 320)",
    )
    parser.add_argument(
        "--max-visual-tokens",
        type=int,
        default=1280,
        help="Maximum number of visual tokens per image",
    )
    parser.add_argument(
        "--peft",
        action="store_true",
        help="Use PEFT/LoRA for parameter-efficient fine-tuning",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=32,
        help="LoRA rank (r parameter)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha parameter",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Per-device training batch size",
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=2,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=200,
        help="Number of warmup steps",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--disable-wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )
    return parser.parse_args()


def create_peft_config(args) -> Optional[LoraConfig]:
    """Create PEFT/LoRA configuration if enabled."""
    if not args.peft:
        return None

    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.1,
        init_lora_weights="gaussian",
        bias="none",
        task_type="FEATURE_EXTRACTION",
        # Target attention projections and custom projection layer
        target_modules=r"(.*(model).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$|.*(custom_text_proj).*$)",
    )


def create_training_args(args) -> TrainingArguments:
    """Create training arguments."""
    return TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        per_device_eval_batch_size=4,
        eval_strategy="steps",
        eval_steps=100,
        dataloader_num_workers=8,
        bf16=True,
        save_steps=500,
        save_total_limit=2,
        logging_steps=10,
        report_to="none" if args.disable_wandb else "wandb",
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        resume_from_checkpoint=args.resume_from_checkpoint,
        remove_unused_columns=False,
    )


def main():
    args = parse_args()

    print("=" * 60)
    print("ColQwen3 Training Script")
    print("=" * 60)
    print(f"Base model: {args.base_model}")
    print(f"Output directory: {args.output_dir}")
    print(f"Embedding dimension: {args.dim}")
    print(f"Max visual tokens: {args.max_visual_tokens}")
    print(f"PEFT enabled: {args.peft}")
    print("=" * 60)

    print_gpu_utilization()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load processor
    print("Loading processor...")
    processor = ColQwen3Processor.from_pretrained(
        args.base_model,
        max_num_visual_tokens=args.max_visual_tokens,
        trust_remote_code=True,
    )

    # Load model
    print("Loading model...")
    model = ColQwen3.from_pretrained(
        args.base_model,
        dim=args.dim,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )

    print_gpu_utilization()

    # Load datasets
    print("Loading training dataset...")
    train_dataset = load_train_set()

    # Create configuration
    config = ColModelTrainingConfig(
        model=model,
        processor=processor,
        train_dataset=train_dataset,
        eval_dataset=None,  # Set to your eval dataset if available
        tr_args=create_training_args(args),
        output_dir=args.output_dir,
        peft_config=create_peft_config(args),
        loss_func=ColbertPairwiseCELoss(),
        run_train=True,
        run_eval=False,  # Set to True if you have eval dataset
    )

    # Create trainer and train
    print("Starting training...")
    training_app = ColModelTraining(config)
    training_app.train()
    training_app.save()

    print("=" * 60)
    print(f"Training complete! Model saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
