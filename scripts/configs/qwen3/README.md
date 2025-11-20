# ColQwen3 Training for RTX 5090

## Setup

Install GPU PyTorch:
```bash
uv pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128 && uv pip install flash-attn --no-build-isolation
```

Test installation:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'Device count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}'); print(f'Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}'); import importlib; fa = importlib.util.find_spec('flash_attn'); print(f'Flash Attention: {\"installed\" if fa else \"not installed\"}')"
```

## Quick Start

### Start Training

```bash
accelerate launch scripts/configs/qwen3/train_colqwen3_model.py \
    --output-dir ./models/my-colqwen3-model \
    --peft \
    --attn-impl flash_attention_2 \
    --wandb-project colqwen3 \
    --run-name my-experiment \
    --resume
```

## Checkpoint Management

Training automatically saves checkpoints every 500 steps. You can:
- Stop training anytime (Ctrl+C)
- Resume later with `--resume`
- See [CHECKPOINT_GUIDE.md](CHECKPOINT_GUIDE.md) for detailed instructions

**Key commands:**
- `--resume` - Resume from last checkpoint
- `--save-steps 250` - Save more frequently (every 250 steps)
- `--resume-from-checkpoint ./path/to/checkpoint` - Resume from specific checkpoint

## Complete Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `--output-dir` | Required | Output directory for model and checkpoints |
| `--base-model` | Qwen/Qwen3-VL-2B-Instruct | Base model to fine-tune |
| `--peft` | False | Use LoRA/PEFT (recommended for RTX 5090) |
| `--attn-impl` | flash_attention_2 | Attention: flash_attention_2, sdpa, or eager |
| `--batch-size` | 8 | Per-device training batch size |
| `--grad-accum-steps` | 4 | Gradient accumulation steps |
| `--num-epochs` | 5 | Number of training epochs |
| `--lr` | 2e-4 | Learning rate |
| `--save-steps` | 500 | Save checkpoint every N steps |
| `--resume` | False | Resume from last checkpoint |
| `--wandb-project` | None | WandB project name for logging |
| `--run-name` | None | WandB run name |

## Tips for RTX 5090

### Maximize Performance

1. **Use Flash Attention 2**
   ```bash
   pip install flash-attn --no-build-isolation
   --attn-impl flash_attention_2 --batch-size 12
   ```

2. **Enable Mixed Precision**
   ```bash
   accelerate launch --mixed_precision=bf16 scripts/configs/qwen3/train_colqwen3_model.py ...
   ```

3. **Optimize Batch Size**
   - Flash Attention: 12-16
   - SDPA: 10-12
   - Eager: 8-10

### Manage Long Training Sessions

1. **Save Frequently**
   ```bash
   --save-steps 250  # Save every 250 steps
   ```

2. **Use WandB**
   ```bash
   --wandb-project colqwen3 --run-name my-experiment
   ```

3. **Monitor GPU**
   ```bash
   # In another terminal
   watch -n 1 nvidia-smi
   ```

4. **Stop/Resume Safely**
   - Press Ctrl+C once (wait for checkpoint save)
   - Resume with `--resume` flag
