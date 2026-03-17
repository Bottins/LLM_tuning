"""
finetune.py
Fine-tuning di Qwen3 con LoRA via Hugging Face Accelerate + FSDP.
Funziona sia in locale (1 GPU) che su cluster multi-nodo (CINECA).

Uso locale (1 GPU):
    python finetune.py --model_id Qwen/Qwen3-1.7B --local

Uso CINECA (lanciato dal job PBS tramite accelerate):
    accelerate launch --config_file accelerate_fsdp.yaml finetune.py \
        --model_id Qwen/Qwen3-4B
"""

import argparse
import json
import os
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    get_cosine_schedule_with_warmup,
)

# ── Argomenti ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-1.7B")
parser.add_argument("--train_file", type=str, default="data/train.jsonl")
parser.add_argument("--val_file", type=str, default="data/val.jsonl")
parser.add_argument("--output_dir", type=str, default="output/sciml-lora")
parser.add_argument("--max_length", type=int, default=1024)
parser.add_argument("--batch_size", type=int, default=2)   # per GPU
parser.add_argument("--grad_accum", type=int, default=8)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--lora_r", type=int, default=16)
parser.add_argument("--lora_alpha", type=int, default=32)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--local", action="store_true",
                    help="Modalità locale: carica in 4-bit per risparmiare VRAM")
args = parser.parse_args()

# ── Setup ──────────────────────────────────────────────────────────────────────
set_seed(args.seed)
accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum)
is_main = accelerator.is_main_process


def log(msg):
    if is_main:
        print(msg, flush=True)


log(f"Accelerator: {accelerator.num_processes} processi, device={accelerator.device}")

# ── Tokenizer ──────────────────────────────────────────────────────────────────
log(f"Carico tokenizer: {args.model_id}")
tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ── Dataset ────────────────────────────────────────────────────────────────────
def load_jsonl(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def tokenize_sample(sample: dict) -> dict:
    """
    Applica il chat template del modello e tokenizza.
    Il label è -100 per i token di sistema/utente (non vengono usati nel loss).
    """
    messages = sample["messages"]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    enc = tokenizer(
        text,
        max_length=args.max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].squeeze(0)
    attention_mask = enc["attention_mask"].squeeze(0)

    # Maschera i token prima dell'ultima risposta dell'assistente
    labels = input_ids.clone()
    # Trova dove inizia la risposta dell'assistente (dopo l'ultimo <|im_start|>assistant)
    assistant_token = tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)
    for i in range(len(input_ids) - len(assistant_token)):
        if input_ids[i : i + len(assistant_token)].tolist() == assistant_token:
            labels[:i + len(assistant_token)] = -100  # ignora prompt nel loss
            break

    labels[attention_mask == 0] = -100  # ignora padding

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


log("Carico e tokenizo il dataset...")
train_raw = load_jsonl(args.train_file)
val_raw = load_jsonl(args.val_file)

train_dataset = Dataset.from_list(train_raw).map(
    tokenize_sample, remove_columns=["messages"]
)
val_dataset = Dataset.from_list(val_raw).map(
    tokenize_sample, remove_columns=["messages"]
)
train_dataset.set_format("torch")
val_dataset.set_format("torch")
log(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

# ── Modello ────────────────────────────────────────────────────────────────────
log(f"Carico modello: {args.model_id}")
load_kwargs = dict(
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
if args.local:
    # Modalità locale: quantizzazione 4-bit per stare nella VRAM della 4070
    from transformers import BitsAndBytesConfig
    load_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    load_kwargs["device_map"] = "auto"
else:
    # CINECA: FSDP gestisce la distribuzione, non usiamo device_map
    load_kwargs["device_map"] = None

model = AutoModelForCausalLM.from_pretrained(args.model_id, **load_kwargs)

# ── LoRA ───────────────────────────────────────────────────────────────────────
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"],
    bias="none",
)
model = get_peft_model(model, lora_config)
if is_main:
    model.print_trainable_parameters()

# ── Ottimizzatore e scheduler ──────────────────────────────────────────────────
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=args.lr,
    weight_decay=0.01,
)
total_steps = (len(train_loader) // args.grad_accum) * args.epochs
warmup_steps = total_steps // 10
scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

# ── Accelerate prepare ─────────────────────────────────────────────────────────
model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
    model, optimizer, train_loader, val_loader, scheduler
)

# ── Training loop ──────────────────────────────────────────────────────────────
Path(args.output_dir).mkdir(parents=True, exist_ok=True)
best_val_loss = float("inf")

for epoch in range(args.epochs):
    model.train()
    total_loss = 0.0
    n_steps = 0

    for step, batch in enumerate(train_loader):
        with accelerator.accumulate(model):
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        n_steps += 1

        if step % 50 == 0:
            log(f"Epoch {epoch+1} | Step {step}/{len(train_loader)} | Loss: {loss.item():.4f}")

    avg_train_loss = total_loss / n_steps
    log(f"── Epoch {epoch+1} train loss: {avg_train_loss:.4f}")

    # ── Validazione ────────────────────────────────────────────────────────────
    model.eval()
    val_loss = 0.0
    n_val = 0
    with torch.no_grad():
        for batch in val_loader:
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            val_loss += outputs.loss.item()
            n_val += 1
    avg_val_loss = val_loss / n_val
    log(f"── Epoch {epoch+1} val   loss: {avg_val_loss:.4f}")

    # ── Salva il miglior checkpoint ────────────────────────────────────────────
    if avg_val_loss < best_val_loss and is_main:
        best_val_loss = avg_val_loss
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(f"{args.output_dir}/best")
        tokenizer.save_pretrained(f"{args.output_dir}/best")
        log(f"   Salvato checkpoint (val_loss={best_val_loss:.4f})")

log("Training completato.")
