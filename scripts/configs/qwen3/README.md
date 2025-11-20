# ColQwen3 Training Configuration

This directory contains training scripts for ColQwen3, a document retrieval model based on Qwen3-VL.

## Overview

ColQwen3 is built on top of the Qwen3-VL vision-language model from Alibaba, incorporating the ColPali late-interaction architecture for efficient multi-vector document retrieval.

### Key Features

- **Base Model**: Qwen3-VL (2B, 4B, or 8B variants)
- **Architecture Improvements**:
  - Interleaved-MRoPE for better positional embeddings
  - DeepStack for finer-grained visual feature fusion
  - Text-Timestamp Alignment for improved grounding
  - Native Dynamic Resolution (up to 28 tokens per image patch)
- **Embedding Dimension**: 128
- **Training Method**: LoRA fine-tuning with ColBERT-style late interaction

## Requirements

- Python >= 3.9
- transformers >= 4.57.0 (required for Qwen3-VL support)
- torch >= 2.2.0
- Install with: `pip install -e ".[train]"`

## Training

### Basic Training Command

```bash
accelerate launch scripts/configs/qwen3/train_colqwen3_model.py \
  --output-dir ./models/colqwen3-v1 \
  --peft
```

### Advanced Training Options

```bash
accelerate launch --multi_gpu scripts/configs/qwen3/train_colqwen3_model.py \
  --output-dir ./models/colqwen3-v1 \
  --base-model-path "Qwen/Qwen3-VL-2B-Instruct" \
  --max-visual-tokens 768 \
  --lr 2e-4 \
  --tau 0.02 \
  --loss pairwise \
  --peft
```

### Arguments

- `--output-dir`: Directory to save the trained model (required)
- `--base-model-path`: Base model to use (default: "Qwen/Qwen3-VL-2B-Instruct")
- `--max-visual-tokens`: Maximum number of visual tokens (default: 768)
- `--lr`: Learning rate (default: 2e-4)
- `--tau`: Temperature for loss function (default: 0.02)
- `--trainer`: Trainer type - "hf" or "torch" (default: "hf")
- `--loss`: Loss function - "ce" or "pairwise" (default: "ce")
- `--peft`: Use LoRA fine-tuning (recommended for efficiency)

### Recommended Settings

For **Qwen3-VL-2B-Instruct**:
- Batch size: 64 per device (with gradient accumulation)
- Learning rate: 2e-4
- LoRA rank: 32
- Max visual tokens: 768-1024
- Loss: pairwise (more stable)
- Flash Attention 2: enabled

For **Qwen3-VL-4B-Instruct**:
- Reduce batch size to 32-48 per device
- Same other hyperparameters

For **Qwen3-VL-8B-Instruct**:
- Reduce batch size to 16-32 per device
- Consider using gradient checkpointing

## Training Data

The script uses the standard ColPali training dataset:
- Training: `vidore/colpali_train_set`
- Evaluation: Local test split

## Evaluation

After training, evaluate on the ViDoRe benchmark:
- InfoVQA, DocVQA, ArxivQA
- TabFQuAD, TatDQA, SyntheticDocQA
- Metrics: NDCG@5, NDCG@10

## Model Architecture

```python
from colpali_engine.models import ColQwen3, ColQwen3Processor

# Load processor
processor = ColQwen3Processor.from_pretrained(
    "Qwen/Qwen3-VL-2B-Instruct",
    max_num_visual_tokens=768
)

# Load model
model = ColQwen3.from_pretrained(
    "Qwen/Qwen3-VL-2B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
)
```

## Troubleshooting

### ImportError for Qwen3VL classes
- Ensure transformers >= 4.57.0: `pip install "transformers>=4.57.0"`
- Verify installation: `python -c "from transformers.models.qwen3_vl import Qwen3VLModel"`

### CUDA Out of Memory
- Reduce batch size: `--per-device-train-batch-size 32`
- Enable gradient checkpointing (already enabled by default)
- Use LoRA: `--peft`
- Reduce max visual tokens: `--max-visual-tokens 512`

### Flash Attention Issues
- Install flash-attn: `pip install flash-attn --no-build-isolation`
- Or disable in code: change `attn_implementation="flash_attention_2"` to `attn_implementation="eager"`

## Expected Performance

ColQwen3 should match or exceed ColQwen2.5 performance due to:
- Better visual feature extraction (DeepStack)
- Improved positional encodings (Interleaved-MRoPE)
- Higher resolution processing capability

Target NDCG@5 on ViDoRe: 75-80+

## Citation

If you use ColQwen3, please cite:

```bibtex
@misc{colpali2024,
  title={ColPali: Efficient Document Retrieval with Vision Language Models},
  author={Faysse, Manuel and Sibille, Hugues and Wu, Tony},
  year={2024}
}

@misc{qwen3vl2025,
  title={Qwen3-VL Technical Report},
  author={Qwen Team},
  year={2025}
}
```
