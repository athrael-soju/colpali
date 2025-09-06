#!/usr/bin/env python3
"""
Evaluation script for ColIntern3.5 on ViDoRe benchmarks v1 and v2.

Usage:
    python scripts/evaluate_colintern3_5.py [--model-path MODEL_PATH] [--batch-size BATCH_SIZE]
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import mteb
import torch
from mteb.model_meta import ModelMeta

# Add the project root to the path so we can import our modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mteb_wrappers.colintern3_5_models import ColIntern3_5Wrapper, colintern3_5_1b_lora

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate ColIntern3.5 on ViDoRe benchmarks")
    parser.add_argument(
        "--model-path",
        type=str,
        default="output/checkpoint-1847",  # Default to our trained checkpoint
        help="Path to the ColIntern3.5 trained checkpoint (must be a PEFT checkpoint)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["ViDoRe(v1)", "ViDoRe(v2)"],
        help="Benchmarks to evaluate on"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evals/colintern3_5-1847",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run evaluation on (default: auto-detect)"
    )
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Validate that model path exists and is a trained checkpoint
    if not os.path.exists(args.model_path):
        logger.error(f"Model path {args.model_path} does not exist!")
        sys.exit(1)
    
    # Validate that this is actually a PEFT checkpoint
    adapter_config_path = os.path.join(args.model_path, "adapter_config.json")
    adapter_model_path = os.path.join(args.model_path, "adapter_model.safetensors")
    
    if not os.path.exists(adapter_config_path):
        logger.error(f"adapter_config.json not found in {args.model_path}. This must be a trained PEFT checkpoint!")
        sys.exit(1)
    if not os.path.exists(adapter_model_path):
        logger.error(f"adapter_model.safetensors not found in {args.model_path}. This must be a trained PEFT checkpoint!")
        sys.exit(1)
    
    logger.info(f"✅ Validated trained checkpoint at: {args.model_path}")
    
    logger.info(f"🚀 Evaluating TRAINED ColIntern3.5 model from: {args.model_path}")
    logger.info(f"📊 Benchmarks: {args.benchmarks}")
    logger.info(f"⚙️  Batch size: {args.batch_size}")
    logger.info(f"💾 Output directory: {args.output_dir}")
    
    # Create custom model metadata for the specific trained checkpoint
    checkpoint_name = Path(args.model_path).name
    model_name = f"athrael-soju/colintern3_5-{checkpoint_name}-trained"
    
    custom_model_meta = ModelMeta(
        loader=lambda name=None: ColIntern3_5Wrapper(
            model_name=args.model_path,
            device=args.device,
            torch_dtype=torch.bfloat16,
        ),
        name=model_name,
        revision=checkpoint_name,
        memory_usage_mb=1800,
        max_tokens=32768,
        embed_dim=3584,
        modalities=["image", "text"],
        framework=["ColPali"],
        similarity_fn_name="max_sim",
        languages=["eng-Latn"],
        release_date="2025-01-09",
        n_parameters=905_000_000,
        license="apache-2.0",
        open_weights=True,
        public_training_code="https://github.com/illuin-tech/colpali",
        public_training_data="https://huggingface.co/datasets/vidore/colpali_train_set",
        reference="https://huggingface.co/OpenGVLab/InternVL3_5-1B-HF",
        use_instructions=True,
        training_datasets={
            "DocVQA": ["train"],
            "InfoVQA": ["train"], 
            "TATDQA": ["train"],
            "arXivQA": ["train"],
        },
    )
    
    # Load the model
    logger.info("🔄 Loading TRAINED model...")
    try:
        model = custom_model_meta.load_model()
        logger.info("✅ TRAINED model loaded successfully!")
        
        # Additional validation that we loaded the trained model
        if hasattr(model, 'mdl') and hasattr(model.mdl, 'custom_text_proj'):
            logger.info("✅ Confirmed: Model has custom_text_proj (trained weights loaded)")
        else:
            logger.error("❌ Error: Model does not have custom_text_proj - this suggests base model was loaded instead of trained!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Failed to load trained model: {e}")
        sys.exit(1)
    
    # Load tasks
    logger.info(f"Loading benchmarks: {args.benchmarks}")
    try:
        tasks = mteb.get_benchmarks(names=args.benchmarks)
        logger.info(f"Loaded {len(tasks)} tasks")
    except Exception as e:
        logger.error(f"Failed to load benchmarks: {e}")
        sys.exit(1)
    
    # Create evaluator
    evaluator = mteb.MTEB(tasks=tasks)
    
    # Run evaluation with overwrite to force re-evaluation of the new model
    logger.info("Starting evaluation...")
    try:
        results = evaluator.run(
            model,
            output_folder=args.output_dir,
            batch_size=args.batch_size,
            verbosity=2,
            overwrite_results=True,  # Force re-evaluation instead of using cached results
        )
        

        logger.info("Evaluation completed successfully!")
        logger.info(f"Results saved to: {args.output_dir}")
        
        # Print summary of results
        if results:
            logger.info("\n" + "="*50)
            logger.info("EVALUATION SUMMARY")
            logger.info("="*50)
            
            # Handle both dict and list result formats
            if isinstance(results, dict):
                for task_name, task_results in results.items():
                    if isinstance(task_results, dict) and 'main_score' in task_results:
                        score = task_results['main_score']
                        logger.info(f"{task_name}: {score:.3f}")
                    else:
                        logger.info(f"{task_name}: {task_results}")
            elif isinstance(results, list):
                for task_result in results:
                    if isinstance(task_result, dict):
                        task_name = task_result.get('task_name', 'Unknown')
                        main_score = task_result.get('main_score', 'N/A')
                        logger.info(f"{task_name}: {main_score}")
                    else:
                        logger.info(f"Result: {task_result}")
            
            logger.info("="*50)
    
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
