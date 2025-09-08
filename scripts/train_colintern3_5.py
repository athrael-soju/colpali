#!/usr/bin/env python3
"""
ColIntern3.5 Training Script - Standalone version

This script trains a ColPali model based on InternVL3.5-1B for document retrieval,
importing models directly from the local implementation.
"""

import argparse
import logging
import os
import shutil
import sys
import warnings
from pathlib import Path

import torch
import wandb
from datasets import load_dataset
from peft import LoraConfig
from transformers import TrainingArguments, set_seed

# Setup paths
_THIS_FILE = Path(__file__).resolve()
_COLPALI_DIR = _THIS_FILE.parents[1]  # Go up one level from scripts to project root
if str(_COLPALI_DIR) not in sys.path:
    sys.path.insert(0, str(_COLPALI_DIR))

# Import our local models
from colpali_engine.models.internvl3_5.colintern3_5 import ColIntern3_5, ColIntern3_5Processor
from colpali_engine.trainer.colmodel_training import ColModelTraining, ColModelTrainingConfig
from colpali_engine.loss.late_interaction_losses import ColbertLoss, ColbertPairwiseCELoss
from colpali_engine.utils.dataset_transformation import load_train_set
from colpali_engine.data.dataset import ColPaliEngineDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("colintern3_5_training.log")
    ]
)
logger = logging.getLogger(__name__)

def check_environment():
    """Check if the environment is ready for training."""
    logger.info("Checking environment...")    
    
    # Check GPU
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. This script requires a CUDA-capable GPU.")
        sys.exit(1)
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    
    # Check for flash attention
    try:
        import flash_attn
        logger.info(f"Flash Attention available: {flash_attn.__version__}")
    except ImportError:
        logger.warning("Flash Attention not available. Consider installing for better performance.")
    return True

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train ColIntern3.5 model")
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./output/colintern3_5",
        help="Output directory for model and logs"
    )
    parser.add_argument(
        "--model-path", 
        type=str, 
        default="./InternVL3_5-1B-HF",
        help="Path to the InternVL3.5 model"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=64,
        help="Per-device training batch size"
    )
    parser.add_argument(
        "--gradient-accumulation-steps", 
        type=int, 
        default=4,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--learning-rate", 
        type=float, 
        default=2e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--num-epochs", 
        type=int, 
        default=1,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--lora-r", 
        type=int, 
        default=32,
        help="LoRA rank"
    )
    parser.add_argument(
        "--lora-alpha", 
        type=int, 
        default=32,
        help="LoRA alpha"
    )
    parser.add_argument(
        "--max-length", 
        type=int, 
        default=50,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--loss-type", 
        type=str, 
        default="colbert",
        choices=["pairwise", "colbert"],
        help="Type of loss function to use"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--wandb-project", 
        type=str, 
        default="colintern3_5",
        help="Weights & Biases project name"
    )
    parser.add_argument(
        "--no-wandb", 
        action="store_true",
        help="Disable Weights & Biases logging"
    )
    parser.add_argument(
        "--max-patches", 
        type=int, 
        default=12,
        help="Maximum number of image patches (InternVL specific, replaces max_num_visual_tokens)"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug mode with reduced training"
    )
    
    return parser.parse_args()

def create_config(args):
    """Create training configuration."""
    # Setup processor
    processor = ColIntern3_5Processor.from_pretrained(
        args.model_path,
        trust_remote_code=True
    )
    
    # Setup model  
    model = ColIntern3_5.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True
    )
    
    # Setup datasets - using HuggingFace dataset format like ColQwen 2.5
    train_dataset = load_train_set()
    eval_dataset = ColPaliEngineDataset(
        load_dataset("./data_dir/colpali_train_set", split="test"),
        pos_target_column_name="image"
    )
    
    # Setup loss function
    if args.loss_type == "pairwise":
        loss_func = ColbertPairwiseCELoss(normalize_scores=False)
    else:
        loss_func = ColbertLoss(
            temperature=0.02,
            normalize_scores=True,
            use_smooth_max=False
        )
    
    # Setup training arguments
    tr_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        per_device_eval_batch_size=max(1, args.batch_size // 2),
        eval_strategy="steps",
        dataloader_num_workers=0,  # Disable multiprocessing on WSL
        dataloader_prefetch_factor=1,
        save_steps=500 if not args.debug else 50,
        logging_steps=10,
        eval_steps=100 if not args.debug else 25,
        warmup_steps=500,
        lr_scheduler_type="linear",
        learning_rate=args.learning_rate,
        save_total_limit=2,
        bf16=True,
        dataloader_pin_memory=False,
        optim="adamw_torch_fused",
        remove_unused_columns=False,
        tf32=True,
        report_to="wandb" if not args.no_wandb else "none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        max_steps=100 if args.debug else -1,
    )
    
    # Setup LoRA configuration
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.1,
        init_lora_weights="gaussian",
        bias="none",
        task_type="FEATURE_EXTRACTION",
        target_modules="(.*(model)(?!.*visual).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$|.*(custom_text_proj).*$)",
    )
    
    # Create training configuration
    config = ColModelTrainingConfig(
        output_dir=args.output_dir,
        processor=processor,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        max_length=args.max_length,
        run_eval=True,
        loss_func=loss_func,
        tr_args=tr_args,
        peft_config=peft_config,
    )
    
    return config

def main():
    """Main training function."""
    args = parse_args()
    
    # Setup environment
    check_environment()
    set_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            config=vars(args)
        )
    
    logger.info(f"Starting ColIntern3.5 training")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    logger.info(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"LoRA rank: {args.lora_r}")
    
    # Test model loading first
    try:
        logger.info("Testing model loading...")
        test_processor = ColIntern3_5Processor.from_pretrained(
            args.model_path,
            trust_remote_code=True
        )
        test_model = ColIntern3_5.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        logger.info("✓ Model loading test successful")
        del test_processor, test_model
        torch.cuda.empty_cache()
    except Exception as e:
        logger.error(f"Model loading test failed: {e}")
        sys.exit(1)
    
    # Create configuration
    try:
        logger.info("Creating training configuration...")
        config = create_config(args)
        logger.info("✓ Configuration created successfully")
    except Exception as e:
        logger.error(f"Failed to create configuration: {e}")
        sys.exit(1)
    
    # Save configuration for reproducibility
    shutil.copy(__file__, output_dir / "training_script.py")
    
    # Initialize trainer
    try:
        logger.info("Initializing trainer...")
        trainer = ColModelTraining(config)
        logger.info("✓ Trainer initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize trainer: {e}")
        sys.exit(1)
    
    # Start training
    try:
        logger.info("Starting training...")
        trainer.train()
        logger.info("✓ Training completed successfully")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)
    
    # Save model
    try:
        logger.info("Saving model...")
        trainer.save()
        logger.info(f"✓ Model saved to: {output_dir}")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        sys.exit(1)
    
    # Finish wandb
    if not args.no_wandb:
        wandb.finish()
    
    logger.info("Training pipeline completed successfully!")
    
    # Print memory usage
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.max_memory_allocated() / (1024**3)
        logger.info(f"Peak GPU memory usage: {memory_allocated:.2f} GB")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
