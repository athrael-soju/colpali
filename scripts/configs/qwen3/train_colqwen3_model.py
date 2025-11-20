import argparse
import os
import shutil
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import TrainingArguments

from colpali_engine.data.dataset import ColPaliEngineDataset
from colpali_engine.loss.late_interaction_losses import ColbertLoss, ColbertPairwiseCELoss
from colpali_engine.models import ColQwen3, ColQwen3Processor
from colpali_engine.trainer.colmodel_torch_training import ColModelTorchTraining
from colpali_engine.trainer.colmodel_training import ColModelTraining, ColModelTrainingConfig
from colpali_engine.utils.dataset_transformation import load_train_set


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=str, required=True, help="where to write model + script copy")
    p.add_argument("--base-model", type=str, default="Qwen/Qwen3-VL-2B-Instruct", help="base model to use")
    p.add_argument("--lr", type=float, default=2e-4, help="learning rate")
    p.add_argument("--tau", type=float, default=0.02, help="temperature for loss function")
    p.add_argument("--trainer", type=str, default="hf", choices=["torch", "hf"], help="trainer to use")
    p.add_argument("--loss", type=str, default="pairwise", choices=["ce", "pairwise"], help="loss function to use")
    p.add_argument("--peft", action="store_true", help="use PEFT for training")
    p.add_argument("--batch-size", type=int, default=18, help="per device train batch size")
    p.add_argument("--grad-accum-steps", type=int, default=2, help="gradient accumulation steps")
    p.add_argument("--num-epochs", type=int, default=5, help="number of training epochs")
    p.add_argument("--max-visual-tokens", type=int, default=1024, help="max number of visual tokens")
    p.add_argument("--eval-batch-size", type=int, default=8, help="per device eval batch size")
    p.add_argument("--use-local-data", action="store_true", help="use local data instead of HuggingFace")
    p.add_argument("--attn-impl", type=str, default="flash_attention_2",
                   choices=["flash_attention_2", "eager", "sdpa"],
                   help="attention implementation to use")
    p.add_argument("--wandb-project", type=str, default=None, help="wandb project name")
    p.add_argument("--run-name", type=str, default=None, help="wandb run name")
    p.add_argument("--resume", action="store_true", help="resume from last checkpoint in output-dir")
    p.add_argument("--resume-from-checkpoint", type=str, default=None, help="specific checkpoint path to resume from")
    p.add_argument("--save-steps", type=int, default=250, help="save checkpoint every N steps")
    p.add_argument("--num-workers", type=int, default=4, help="number of dataloader workers")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Print GPU configuration
    print("\n" + "="*60)
    print("CUDA/GPU Configuration:")
    print("="*60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    else:
        print("WARNING: CUDA is not available! Training will run on CPU (very slow)")
    print("="*60 + "\n")

    # Set environment variable for dataset loading
    if not args.use_local_data:
        os.environ["USE_LOCAL_DATASET"] = "0"
        print("Using HuggingFace datasets")
    else:
        os.environ["USE_LOCAL_DATASET"] = "1"
        print("Using local datasets")

    # Set WandB environment variables if specified
    if args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project
        if args.run_name:
            os.environ["WANDB_RUN_NAME"] = args.run_name

    # Configure loss function
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

    # Load eval dataset
    if args.use_local_data:
        eval_dataset = ColPaliEngineDataset(
            load_dataset("./data_dir/colpali_train_set", split="test"),
            pos_target_column_name="image"
        )
    else:
        # Load the dataset and filter out None query values
        raw_eval_dataset = load_dataset("vidore/syntheticDocQA_energy_test", split="test")
        # Filter out examples with None queries
        filtered_eval_dataset = raw_eval_dataset.filter(lambda ex: ex["query"] is not None)
        print(f"Filtered eval dataset: {len(raw_eval_dataset)} -> {len(filtered_eval_dataset)} examples (removed {len(raw_eval_dataset) - len(filtered_eval_dataset)} None queries)")
        eval_dataset = ColPaliEngineDataset(
            filtered_eval_dataset,
            pos_target_column_name="image"
        )

    # Configure training
    config = ColModelTrainingConfig(
        output_dir=args.output_dir,
        processor=ColQwen3Processor.from_pretrained(
            pretrained_model_name_or_path=args.base_model,
            max_num_visual_tokens=args.max_visual_tokens,
        ),
        model=ColQwen3.from_pretrained(
            pretrained_model_name_or_path=args.base_model,
            torch_dtype=torch.bfloat16,
            attn_implementation=args.attn_impl,
        ),
        train_dataset=load_train_set(),
        eval_dataset=eval_dataset,
        run_eval=True,
        loss_func=loss_func,
        tr_args=TrainingArguments(
            output_dir=None,
            overwrite_output_dir=not (args.resume or args.resume_from_checkpoint),
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum_steps,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            per_device_eval_batch_size=args.eval_batch_size,
            eval_strategy="steps",
            dataloader_num_workers=args.num_workers,
            save_steps=args.save_steps,
            logging_steps=10,
            eval_steps=100,
            warmup_steps=100,
            learning_rate=args.lr,
            save_total_limit=1,
            report_to="wandb" if args.wandb_project else "none",
            resume_from_checkpoint=args.resume_from_checkpoint if args.resume_from_checkpoint else (True if args.resume else None),
        ),
        peft_config=LoraConfig(
            r=32,
            lora_alpha=32,
            lora_dropout=0.1,
            init_lora_weights="gaussian",
            bias="none",
            task_type="FEATURE_EXTRACTION",
            target_modules="(.*(model).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$|.*(custom_text_proj).*$)",
        )
        if args.peft
        else None,
    )

    # Make sure output_dir exists and copy script for provenance
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(__file__), Path(config.output_dir) / Path(__file__).name)

    print(f"\n{'='*60}")
    print("Training Configuration:")
    print(f"{'='*60}")
    print(f"Base model: {args.base_model}")
    print(f"Output directory: {args.output_dir}")
    print(f"Learning rate: {args.lr}")
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient accumulation steps: {args.grad_accum_steps}")
    print(f"Effective batch size: {args.batch_size * args.grad_accum_steps}")
    print(f"Loss function: {args.loss}")
    print(f"Using PEFT: {args.peft}")
    print(f"Attention implementation: {args.attn_impl}")
    print(f"Max visual tokens: {args.max_visual_tokens}")
    print(f"Save checkpoints every: {args.save_steps} steps")
    if args.resume or args.resume_from_checkpoint:
        print(f"Resuming from checkpoint: {args.resume_from_checkpoint if args.resume_from_checkpoint else 'last checkpoint in output-dir'}")
    print(f"{'='*60}\n")

    trainer = ColModelTraining(config) if args.trainer == "hf" else ColModelTorchTraining(config)
    trainer.train()
    trainer.save()

    print(f"\n{'='*60}")
    print("Training completed successfully!")
    print(f"Model saved to: {args.output_dir}")
    print(f"{'='*60}\n")
