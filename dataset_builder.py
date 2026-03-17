"""
dataset_builder.py
Genera un dataset Q&A su PDE e PINNs in formato conversazionale
compatibile con il fine-tuning di LLM (stile chat template).

Output: data/sciml_qa.jsonl
"""

import json
import random
import os
from pathlib import Path

# ── Seed per riproducibilità ──────────────────────────────────────────────────
random.seed(42)

# ── Template Q&A ─────────────────────────────────────────────────────────────
# Ogni entry è un dict con "question" e "answer".
# Vengono poi convertiti in chat format (system / user / assistant).

SYSTEM_PROMPT = (
    "You are an expert in Scientific Machine Learning, Physics-Informed Neural Networks "
    "(PINNs), and numerical methods for partial differential equations (PDEs). "
    "Provide rigorous, concise, and technically accurate answers."
)

QA_PAIRS = [
    # ── Fondamenti PDE ────────────────────────────────────────────────────────
    {
        "question": "What is a partial differential equation (PDE) and how does it differ from an ODE?",
        "answer": (
            "A partial differential equation (PDE) involves an unknown function of multiple "
            "independent variables and its partial derivatives. For example, the heat equation "
            "∂u/∂t = α ∂²u/∂x² describes how temperature u evolves in space x and time t. "
            "An ordinary differential equation (ODE) involves a function of a single variable "
            "and its ordinary derivatives. PDEs are generally harder to solve because the "
            "solution lives in a higher-dimensional domain and boundary/initial conditions "
            "must be specified on that domain's boundary."
        ),
    },
    {
        "question": "What are the main types of second-order linear PDEs and how are they classified?",
        "answer": (
            "Second-order linear PDEs are classified by the sign of the discriminant B²-4AC "
            "of the principal part Au_xx + Bu_xy + Cu_yy:\n"
            "- Elliptic (B²-4AC < 0): e.g. Laplace equation ∇²u = 0, steady-state problems.\n"
            "- Parabolic (B²-4AC = 0): e.g. heat equation ∂u/∂t = α∇²u, diffusion processes.\n"
            "- Hyperbolic (B²-4AC > 0): e.g. wave equation ∂²u/∂t² = c²∇²u, wave propagation.\n"
            "This classification determines the appropriate numerical method and boundary "
            "condition structure."
        ),
    },
    {
        "question": "What are boundary conditions and why are they essential for PDE problems?",
        "answer": (
            "Boundary conditions (BCs) constrain the solution of a PDE on the boundary ∂Ω "
            "of the domain Ω. Without BCs, a PDE has infinitely many solutions. "
            "The three main types are:\n"
            "- Dirichlet: u = g on ∂Ω (prescribed value).\n"
            "- Neumann: ∂u/∂n = g on ∂Ω (prescribed normal derivative).\n"
            "- Robin: αu + β∂u/∂n = g on ∂Ω (mixed condition).\n"
            "For time-dependent PDEs, initial conditions (ICs) u(x,0) = u₀(x) are also required. "
            "Together, BCs and ICs form a well-posed problem (existence, uniqueness, stability)."
        ),
    },
    {
        "question": "What is the Gray-Scott reaction-diffusion system and what phenomena does it model?",
        "answer": (
            "The Gray-Scott system models autocatalytic chemical reactions between two species "
            "u and v on a domain Ω:\n"
            "  ∂u/∂t = Dᵤ∇²u − uv² + F(1−u)\n"
            "  ∂v/∂t = D_v∇²v + uv² − (F+K)v\n"
            "where Dᵤ, D_v are diffusion coefficients, F is the feed rate, and K is the kill rate. "
            "The nonlinear term uv² models the autocatalytic reaction U + 2V → 3V. "
            "Depending on (F,K), the system produces rich Turing-like patterns: spots, stripes, "
            "labyrinths, and spatiotemporal chaos. It is a standard benchmark for inverse "
            "parameter identification in Scientific Machine Learning."
        ),
    },
    {
        "question": "What is the Richards equation and why is it challenging to solve numerically?",
        "answer": (
            "The Richards equation models water flow in variably saturated porous media:\n"
            "  ∂θ/∂t = ∇·[K(h)∇(h+z)] \n"
            "where θ is volumetric water content, h is pressure head, K(h) is the unsaturated "
            "hydraulic conductivity, and z is elevation. The challenges are:\n"
            "1. Strong nonlinearity: K(h) and θ(h) (via van Genuchten relations) vary by orders "
            "   of magnitude.\n"
            "2. Sharp fronts: wetting fronts create steep gradients requiring fine discretization.\n"
            "3. Mixed form: coupling between θ and h causes mass conservation issues.\n"
            "PINNs have been proposed as mesh-free alternatives that embed the PDE directly "
            "in the loss function."
        ),
    },

    # ── PINNs ─────────────────────────────────────────────────────────────────
    {
        "question": "What is a Physics-Informed Neural Network (PINN) and how does it work?",
        "answer": (
            "A PINN (Raissi et al., 2019) is a neural network trained to satisfy both observed "
            "data and the governing PDE simultaneously. Given a PDE F[u]=0 on Ω and boundary "
            "conditions B[u]=g on ∂Ω, a network û_θ(x,t) is optimized to minimize:\n"
            "  L = λ_data · L_data + λ_pde · L_pde + λ_bc · L_bc\n"
            "where L_data measures misfit to observations, L_pde penalizes PDE residuals at "
            "collocation points, and L_bc enforces boundary/initial conditions. "
            "Derivatives are computed via automatic differentiation (autograd). "
            "PINNs are mesh-free and naturally handle irregular geometries and noisy data."
        ),
    },
    {
        "question": "What is the inverse problem in the context of PINNs?",
        "answer": (
            "In a PINN inverse problem, both the network weights θ and unknown PDE parameters "
            "φ (e.g., diffusion coefficients, reaction rates) are optimized jointly. "
            "Given observations {u*(xᵢ,tᵢ)}, the loss becomes:\n"
            "  L(θ,φ) = L_data(θ) + λ · L_pde(θ,φ)\n"
            "The PDE residual acts as a physics regularizer that constrains the search for φ. "
            "This is more data-efficient than purely data-driven approaches because the PDE "
            "structure reduces the effective degrees of freedom. For Gray-Scott, one jointly "
            "estimates F and K from partial observations of u and v fields."
        ),
    },
    {
        "question": "What are the main failure modes of PINNs and how can they be mitigated?",
        "answer": (
            "Common PINN failure modes and mitigations:\n"
            "1. Loss imbalance: λ_pde and λ_data have different magnitudes → use adaptive "
            "   weighting (NTK-based, uncertainty weighting, or manual tuning).\n"
            "2. Spectral bias: networks learn low frequencies first, miss sharp features → "
            "   use Fourier feature embeddings or multi-scale architectures.\n"
            "3. Stiff PDEs: gradients explode or vanish → use causal training (respect "
            "   temporal causality) or domain decomposition.\n"
            "4. Collocation point distribution: uniform sampling misses important regions → "
            "   use residual-based adaptive sampling (RAR).\n"
            "5. Optimization landscape: non-convex loss with many saddle points → use "
            "   learning rate schedules, L-BFGS after Adam warm-up."
        ),
    },
    {
        "question": "What are Fourier feature embeddings and why do they help PINNs?",
        "answer": (
            "Fourier feature embeddings (Tancik et al., 2020) map input coordinates through "
            "a random Fourier basis before the network:\n"
            "  γ(x) = [sin(2πBx), cos(2πBx)]  where B ~ N(0,σ²)\n"
            "This overcomes the spectral bias of standard MLPs, which preferentially learn "
            "low-frequency components. For PINNs solving high-frequency or multi-scale PDEs "
            "(e.g., wave equations, reaction-diffusion with fine patterns), Fourier features "
            "dramatically improve convergence speed and solution accuracy. The bandwidth σ "
            "controls the frequency range captured and is a key hyperparameter."
        ),
    },
    {
        "question": "How does automatic differentiation enable PINNs to compute PDE residuals?",
        "answer": (
            "Automatic differentiation (AD) computes exact derivatives of the network output "
            "with respect to its inputs using the chain rule, without finite differences. "
            "In PyTorch, this is done via torch.autograd.grad with create_graph=True, which "
            "builds a computational graph through the derivative, enabling higher-order "
            "derivatives. For a PINN approximating u(x,t), the PDE residual ∂u/∂t - α∂²u/∂x² "
            "is computed as:\n"
            "  u_t = grad(u, t, create_graph=True)\n"
            "  u_x = grad(u, x, create_graph=True)\n"
            "  u_xx = grad(u_x, x)\n"
            "  residual = u_t - alpha * u_xx\n"
            "This exact differentiation is a key advantage of PINNs over finite element methods."
        ),
    },

    # ── GNN e ottimizzatori ───────────────────────────────────────────────────
    {
        "question": "What is a Graph Neural Network (GNN) and what makes it suitable for mesh-based PDE solving?",
        "answer": (
            "A GNN operates on graph-structured data G=(V,E) by passing messages between "
            "neighboring nodes. For PDE solving on meshes, nodes represent spatial points "
            "and edges represent connectivity (e.g., FEM mesh edges). The message passing:\n"
            "  h_v^{l+1} = UPDATE(h_v^l, AGGREGATE({h_u^l : u∈N(v)}))\n"
            "naturally respects the local structure of PDE operators. GNNs generalize across "
            "mesh resolutions and geometries, unlike fixed-grid CNNs. Models like GNNs in "
            "MeshGraphNets (DeepMind) have shown strong performance on fluid dynamics and "
            "structural mechanics simulations."
        ),
    },
    {
        "question": "What is the Adam optimizer and what are its key hyperparameters?",
        "answer": (
            "Adam (Kingma & Ba, 2015) is an adaptive gradient optimizer combining momentum "
            "and RMSProp. Updates are:\n"
            "  m_t = β₁m_{t-1} + (1-β₁)g_t          (1st moment)\n"
            "  v_t = β₂v_{t-1} + (1-β₂)g_t²          (2nd moment)\n"
            "  m̂_t = m_t/(1-β₁^t),  v̂_t = v_t/(1-β₂^t)  (bias correction)\n"
            "  θ_{t+1} = θ_t − η · m̂_t / (√v̂_t + ε)\n"
            "Key hyperparameters: learning rate η (default 1e-3), β₁=0.9, β₂=0.999, ε=1e-8. "
            "Adam is the default choice for PINNs and LLM fine-tuning due to its robustness "
            "to gradient scale variations."
        ),
    },
    {
        "question": "What is LoRA and why is it used for LLM fine-tuning?",
        "answer": (
            "LoRA (Low-Rank Adaptation, Hu et al., 2022) fine-tunes LLMs by adding trainable "
            "low-rank matrices to frozen pre-trained weights:\n"
            "  W' = W + ΔW = W + BA  where B∈R^{d×r}, A∈R^{r×k}, r≪min(d,k)\n"
            "Only A and B are trained, reducing trainable parameters by 10-10000x. "
            "For a 7B model, full fine-tuning requires ~28GB VRAM; LoRA with r=16 requires "
            "~8GB. This makes fine-tuning feasible on consumer GPUs. LoRA matrices are "
            "typically applied to attention weight matrices (Q,K,V,O projections). "
            "After training, LoRA weights can be merged into W for zero-overhead inference."
        ),
    },
    {
        "question": "What is FSDP and how does it enable training large models across multiple GPUs?",
        "answer": (
            "Fully Sharded Data Parallel (FSDP, Zhao et al., 2023) shards model parameters, "
            "gradients, and optimizer states across all GPUs, unlike DDP which replicates the "
            "full model on each device. Each GPU holds only 1/N of the parameters. During "
            "forward/backward passes, FSDP performs all-gather to reconstruct full layers "
            "temporarily, then re-shards after computation. This reduces per-GPU memory from "
            "O(M) to O(M/N) for M parameters and N GPUs. FSDP is essential for models that "
            "don't fit on a single GPU and is the standard for multi-node LLM training in "
            "PyTorch. It integrates with Hugging Face Accelerate via --fsdp flags."
        ),
    },

    # ── Codice PyTorch ────────────────────────────────────────────────────────
    {
        "question": "Write a minimal PyTorch implementation of a PINN for the 1D heat equation.",
        "answer": (
            "```python\n"
            "import torch\n"
            "import torch.nn as nn\n\n"
            "class PINN(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            "        self.net = nn.Sequential(\n"
            "            nn.Linear(2, 64), nn.Tanh(),\n"
            "            nn.Linear(64, 64), nn.Tanh(),\n"
            "            nn.Linear(64, 1)\n"
            "        )\n\n"
            "    def forward(self, x, t):\n"
            "        return self.net(torch.cat([x, t], dim=1))\n\n"
            "def pde_residual(model, x, t, alpha=0.01):\n"
            "    x.requires_grad_(True); t.requires_grad_(True)\n"
            "    u = model(x, t)\n"
            "    u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]\n"
            "    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]\n"
            "    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]\n"
            "    return u_t - alpha * u_xx  # heat equation residual\n\n"
            "model = PINN()\n"
            "optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)\n"
            "for step in range(5000):\n"
            "    x_c = torch.rand(256, 1)  # collocation points\n"
            "    t_c = torch.rand(256, 1)\n"
            "    loss_pde = pde_residual(model, x_c, t_c).pow(2).mean()\n"
            "    optimizer.zero_grad(); loss_pde.backward(); optimizer.step()\n"
            "```"
        ),
    },
    {
        "question": "How do you implement LoRA fine-tuning with Hugging Face PEFT for a causal LLM?",
        "answer": (
            "```python\n"
            "from transformers import AutoModelForCausalLM, AutoTokenizer\n"
            "from peft import LoraConfig, get_peft_model, TaskType\n\n"
            "model_id = 'Qwen/Qwen3-1.7B'\n"
            "tokenizer = AutoTokenizer.from_pretrained(model_id)\n"
            "model = AutoModelForCausalLM.from_pretrained(\n"
            "    model_id, torch_dtype=torch.bfloat16, device_map='auto'\n"
            ")\n\n"
            "lora_config = LoraConfig(\n"
            "    task_type=TaskType.CAUSAL_LM,\n"
            "    r=16,           # rank\n"
            "    lora_alpha=32,  # scaling\n"
            "    lora_dropout=0.05,\n"
            "    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],\n"
            "    bias='none',\n"
            ")\n"
            "model = get_peft_model(model, lora_config)\n"
            "model.print_trainable_parameters()\n"
            "# Output: trainable params: ~4M || all params: ~1.7B || trainable%: ~0.24%\n"
            "```"
        ),
    },
]


def build_chat_sample(qa: dict, system: str) -> dict:
    """Converte una coppia Q&A in formato chat messages."""
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": qa["question"]},
            {"role": "assistant", "content": qa["answer"]},
        ]
    }


def save_dataset(pairs: list, output_path: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for qa in pairs:
            sample = build_chat_sample(qa, SYSTEM_PROMPT)
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"Salvato {len(pairs)} campioni in {output_path}")


def augment_pairs(pairs: list, n_augment: int = 3) -> list:
    """
    Augmentation semplice: riformula la domanda con varianti.
    In produzione si può usare un LLM per generare varianti più ricche.
    """
    question_prefixes = [
        "Can you explain ",
        "Could you describe ",
        "Please explain ",
        "What do you know about ",
        "Give me a technical explanation of ",
    ]
    augmented = list(pairs)
    for qa in pairs:
        for _ in range(n_augment):
            prefix = random.choice(question_prefixes)
            new_q = prefix + qa["question"][0].lower() + qa["question"][1:]
            augmented.append({"question": new_q, "answer": qa["answer"]})
    return augmented


if __name__ == "__main__":
    print(f"Campioni base: {len(QA_PAIRS)}")
    all_pairs = augment_pairs(QA_PAIRS, n_augment=2)
    random.shuffle(all_pairs)
    print(f"Campioni totali dopo augmentation: {len(all_pairs)}")

    # Split train/val 90/10
    split = int(0.9 * len(all_pairs))
    train_pairs = all_pairs[:split]
    val_pairs = all_pairs[split:]

    save_dataset(train_pairs, "data/train.jsonl")
    save_dataset(val_pairs, "data/val.jsonl")
    print("Dataset pronto.")
