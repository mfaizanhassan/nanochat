# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

nanochat is a minimal, full-stack implementation of an LLM like ChatGPT designed to run on a single 8XH100 node. It includes the complete pipeline: tokenization, pretraining, finetuning, evaluation, inference, and web serving.

## Development Commands

### Environment Setup
```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate virtual environment
uv venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
uv sync
```

### Building
```bash
# Install Rust/Cargo (required for tokenizer)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# Build the Rust tokenizer
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
```

### Testing
```bash
# Run tests (primarily tokenizer tests)
python -m pytest tests/test_rustbpe.py -v -s

# Run all tests
python -m pytest tests/ -v -s

# Run without slow tests
python -m pytest tests/ -v -s -m "not slow"
```

### Training Pipeline Scripts
Use `torchrun --standalone --nproc_per_node=8` for multi-GPU training:

```bash
# Tokenizer training
python -m scripts.tok_train --max_chars=2000000000
python -m scripts.tok_eval

# Base model pretraining
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=20
torchrun --standalone --nproc_per_node=8 -m scripts.base_loss
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval

# Midtraining and supervised finetuning
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval

# Optional reinforcement learning
torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl
```

### Inference and Chat
```bash
# CLI chat interface
python -m scripts.chat_cli -p "Why is the sky blue?"

# Web UI (ChatGPT-like interface)
python -m scripts.chat_web

# Multi-GPU web serving
python -m scripts.chat_web --num-gpus 4
```

### Complete Training Pipeline
```bash
# Full $100 speedrun (4 hours on 8XH100)
bash speedrun.sh

# Run in screen session with logging
screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh

# With wandb logging
WANDB_RUN=speedrun bash speedrun.sh
```

## Architecture

### Core Components

**nanochat/ module**: Core training and inference engine
- `gpt.py`: Transformer model implementation
- `engine.py`: Training engine with gradient accumulation and distributed training
- `tokenizer.py`: Python wrapper for Rust BPE tokenizer
- `dataloader.py`: Efficient data loading for training
- `dataset.py`: Dataset downloading and preprocessing
- `checkpoint_manager.py`: Model checkpoint saving/loading
- `configurator.py`: Configuration management
- `report.py`: Training metrics and evaluation reporting

**scripts/ directory**: Executable training and inference scripts
- `base_train.py`: Pretraining script
- `mid_train.py`: Midtraining (conversation formatting)
- `chat_sft.py`: Supervised finetuning
- `chat_rl.py`: Reinforcement learning
- `chat_web.py`: FastAPI web server with data parallelism
- `chat_cli.py`: Command-line chat interface
- Evaluation scripts: `base_eval.py`, `chat_eval.py`

**rustbpe/**: Rust-based BPE tokenizer for performance
- Built with maturin/PyO3 for Python bindings
- Significantly faster than pure Python implementations

### Training Pipeline Flow

1. **Tokenizer**: Train BPE tokenizer on ~2B characters
2. **Base Training**: Pretrain transformer on ~54B characters using Chinchilla scaling
3. **Midtraining**: Adapt model to conversation format and special tokens
4. **Supervised Finetuning**: Fine-tune on conversational data
5. **Reinforcement Learning**: Optional RL training on specific tasks like GSM8K

### Model Scaling

Models are defined by depth parameter:
- `d20` (default): 561M parameters, ~$100 to train
- `d26`: ~$300 to train, GPT-2 level performance
- Larger models require adjusting `device_batch_size` to fit in GPU memory

### Key Configuration

- **Multi-GPU**: Uses `torchrun` for distributed training
- **Memory Management**: Adjust `--device_batch_size` (32→16→8→4→2→1) if OOM
- **Data**: Downloads training shards automatically, ~250M chars per shard
- **Evaluation**: Uses CORE metric and standard benchmarks (ARC, GSM8K, HumanEval, MMLU)

### Environment Variables

- `NANOCHAT_BASE_DIR`: Base directory for artifacts (default: `~/.cache/nanochat`)
- `WANDB_RUN`: Wandb run name for logging (use "dummy" to disable)
- `OMP_NUM_THREADS=1`: Recommended for multi-GPU training

## Web Interface

The FastAPI web server (`scripts/chat_web.py`) provides:
- GET `/`: ChatGPT-like web UI
- POST `/chat/completions`: Streaming chat API
- GET `/health`: Health check and worker status
- GET `/stats`: GPU utilization and statistics

Built-in abuse prevention with message limits and parameter clamping.