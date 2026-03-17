# Scientific Machine Learning LLM Fine-Tuning

## Overview

A production-ready pipeline for fine-tuning large language models (Qwen3) on domain-specific Scientific Machine Learning (SciML) Q&A datasets. This system supports both single-GPU local development and multi-node distributed training at HPC scale, enabling efficient adaptation of foundation models to technical scientific domains.

## Key Features

- **Domain Specialization**: Q&A dataset covering PDEs, Physics-Informed Neural Networks (PINNs), Graph Neural Networks (GNNs), and numerical optimization
- **Flexible Training Modes**:
  - Local single-GPU training with 4-bit quantization (QLoRA)
  - Multi-node distributed training with Fully Sharded Data Parallel (FSDP) for 16+ GPUs
- **Production-Grade Infrastructure**: HPC-ready with PBS job scripts for CINECA supercomputing cluster
- **Parameter-Efficient Fine-Tuning**: LoRA (Low-Rank Adaptation) for memory-efficient training
- **Automated Dataset Generation**: Synthetic Q&A pair construction from scientific literature

## Technical Stack

- **Base Model**: Qwen3 (1.7B - 7B parameters)
- **Fine-Tuning Method**: QLoRA (4-bit) / LoRA (16-bit)
- **Distributed Training**: Hugging Face Accelerate with FSDP
- **Quantization**: BitsAndBytes (4-bit NF4)
- **Optimization**: AdamW with cosine learning rate scheduling
- **HPC Integration**: PBS job scheduler (CINECA compatibility)

## Project Structure

```
LLM_tuning/
├── dataset_builder.py       # Generates train/validation Q&A datasets
├── finetune.py              # Main training script (LoRA/FSDP)
├── accelerate_fsdp.yaml     # Accelerate configuration for multi-node FSDP
├── job.pbs                  # PBS job script for CINECA (2 nodes, 16 GPUs)
├── requirements.txt         # Python dependencies
└── data/                    # Generated datasets (train.json, val.json)
    ├── train.json
    └── val.json
```

## Installation

```bash
pip install -r requirements.txt
```

**Requirements**:
- Python ≥ 3.10
- PyTorch ≥ 2.0
- CUDA ≥ 11.8 (for GPU training)
- Transformers, PEFT, Accelerate, BitsAndBytes

## Usage

### Local Training (Single GPU)

```bash
# Step 1: Generate synthetic Q&A dataset
python dataset_builder.py

# Step 2: Fine-tune with 4-bit quantization
python finetune.py --model_id Qwen/Qwen3-1.7B --local
```

**Arguments**:
- `--model_id`: Hugging Face model identifier (e.g., `Qwen/Qwen3-1.7B`)
- `--local`: Enables 4-bit quantization for single-GPU training
- `--epochs`: Number of training epochs (default: 3)
- `--batch_size`: Per-device batch size (default: 4)
- `--learning_rate`: Peak learning rate (default: 2e-4)

### Multi-Node Distributed Training (HPC)

For CINECA or other PBS-based HPC systems:

```bash
# Step 1: Generate dataset (on login node or compute node)
python dataset_builder.py

# Step 2: Submit PBS job for 2-node, 16-GPU training
qsub job.pbs

# Step 3: Monitor job status
qstat -u $USER
```

**Job Configuration** (`job.pbs`):
- Nodes: 2
- GPUs per node: 8 (total 16 GPUs)
- FSDP Strategy: `FULL_SHARD` with mixed precision (FP16)
- Gradient checkpointing enabled for memory efficiency

## Dataset Format

The `dataset_builder.py` script generates Q&A pairs in the following JSON structure:

```json
{
  "instruction": "Explain the role of physics-informed loss functions in PINNs.",
  "input": "",
  "output": "Physics-informed loss functions constrain neural network training by incorporating PDE residuals as regularization terms, ensuring solutions satisfy governing equations..."
}
```

**Dataset Domains**:
- Partial Differential Equations (PDEs)
- Physics-Informed Neural Networks (PINNs)
- Graph Neural Networks (GNNs)
- Numerical optimization methods
- Computational fluid dynamics

## Training Configuration

### LoRA Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Rank (`r`) | 16 | Low-rank decomposition dimension |
| Alpha (`lora_alpha`) | 32 | Scaling factor for LoRA updates |
| Dropout | 0.05 | Regularization dropout rate |
| Target Modules | `q_proj`, `v_proj` | Attention layers to adapt |

### FSDP Settings (Multi-Node)

| Setting | Value |
|---------|-------|
| Sharding Strategy | `FULL_SHARD` |
| Mixed Precision | FP16 |
| Gradient Checkpointing | Enabled |
| Offload to CPU | Disabled (GPU-only) |

## Performance Benchmarks

| Setup | Throughput | Memory | Training Time (3 epochs) |
|-------|------------|--------|--------------------------|
| 1× A100 (40GB) QLoRA | ~15 samples/sec | 28GB | ~4 hours |
| 16× A100 (FSDP) | ~180 samples/sec | 35GB/GPU | ~30 minutes |

## Output Artifacts

After training, the following artifacts are saved:

- `models/qwen3-sciml/` - Fine-tuned model checkpoints
- `logs/training_metrics.csv` - Loss, learning rate, gradient norms per step
- `configs/lora_config.json` - LoRA adapter configuration

## Use Cases

- **Scientific AI Assistants**: Domain-specific chatbots for research labs
- **Educational Tools**: Interactive tutoring systems for SciML courses
- **Research Acceleration**: Automated literature review and methodology suggestions
- **Simulation Guidance**: Natural language interfaces for PDE solvers and GNN frameworks

## Research Profile

- **Keywords**: LLM fine-tuning, scientific machine learning, QLoRA, FSDP, distributed training, domain adaptation, physics-informed AI
- **Application Domains**: Computational physics, materials science, climate modeling, drug discovery
- **Open Source**: Reproducible training pipeline for academic and industrial research

## Citation

If using this pipeline in academic work, please cite:

```bibtex
@software{bottini2026sciml_llm,
  author = {Bottini, Alessandro},
  title = {Scientific Machine Learning LLM Fine-Tuning Pipeline},
  year = {2026},
  url = {https://github.com/Bottins/LLM_tuning}
}
```

## Acknowledgments

- **Infrastructure**: CINECA supercomputing resources
- **Base Model**: Qwen Team (Alibaba Cloud)
- **Frameworks**: Hugging Face Transformers, PEFT, Accelerate

## License

Open-source for research and educational purposes. Commercial deployment of fine-tuned models should comply with Qwen3 license terms.

---

**Author**: Alessandro Bottini
**Last Updated**: March 2026
**Repository**: Coming soon to GitHub
