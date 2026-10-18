import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoTokenizer,
    Gemma3ForConditionalGeneration,
    Trainer,
    TrainingArguments,
    set_seed,
)

set_seed(42)
folder = Path.home() / "pathology-simplifier"
model_id = "google/medgemma-4b-it"

assert torch.cuda.is_available(), "A GPU is required."

tokenizer = AutoTokenizer.from_pretrained(
    model_id, local_files_only=True
)
tokenizer.padding_side = "right"
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

instruction = (
    "Rewrite the following medical caption in plain English. "
    "Use short sentences and everyday words. "
    "Preserve meaning, negation, and uncertainty. "
    "Do not add medical claims or independently diagnose. "
    "Return only the simplified caption."
)

def prepare(split):
    path = folder / "data_split" / f"{split}.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    examples = []

    for record in records:
        source = record["technical_caption"].strip()
        target = record["simplified_caption"].strip()
        assert source and target, "Empty caption found."

        messages = [{
            "role": "user",
            "content": instruction + "\n\nCaption:\n" + source,
        }]

        prefix = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        full = tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": target}],
            tokenize=True,
            add_generation_prompt=False,
        )

        assert full[:len(prefix)] == prefix, "Chat template mismatch."
        assert len(full) <= 1024, "Example exceeds 1024 tokens."

        examples.append({
            "input_ids": full,
            "attention_mask": [1] * len(full),
            "labels": [-100] * len(prefix) + full[len(prefix):],
        })

    print(f"{split}: {len(examples)} examples", flush=True)
    return Dataset.from_list(examples)

train_data = prepare("train")
validation_data = prepare("validation")

def collate(examples):
    length = max(len(e["input_ids"]) for e in examples)
    batch = {"input_ids": [], "attention_mask": [], "labels": []}

    for example in examples:
        padding = length - len(example["input_ids"])
        batch["input_ids"].append(
            example["input_ids"] + [tokenizer.pad_token_id] * padding
        )
        batch["attention_mask"].append(
            example["attention_mask"] + [0] * padding
        )
        batch["labels"].append(
            example["labels"] + [-100] * padding
        )

    return {
        key: torch.tensor(value, dtype=torch.long)
        for key, value in batch.items()
    }

model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id,
    local_files_only=True,
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
)
model.config.use_cache = False
model.config.text_config.use_cache = False

targets = [
    name for name, module in model.named_modules()
    if "language_model" in name
    and name.rsplit(".", 1)[-1] in {"q_proj", "v_proj"}
    and isinstance(module, torch.nn.Linear)
]
assert targets, "No language-model LoRA targets found."

model = get_peft_model(
    model,
    LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=targets,
        bias="none",
    ),
)
model.print_trainable_parameters()

args = TrainingArguments(
    output_dir=str(folder / "medgemma_checkpoints"),
    num_train_epochs=2,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    warmup_ratio=0.1,
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    logging_steps=5,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",
    remove_unused_columns=False,
    seed=42,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_data,
    eval_dataset=validation_data,
    data_collator=collate,
)

print("Measuring validation loss before training...", flush=True)
before = trainer.evaluate()

trainer.train()

after = trainer.evaluate()
destination = folder / "medgemma_simplifier_adapter"
trainer.save_model(str(destination))
tokenizer.save_pretrained(str(destination))
(destination / "instruction.txt").write_text(
    instruction, encoding="utf-8"
)

metrics = {"before": before, "after": after}
(destination / "validation_metrics.json").write_text(
    json.dumps(metrics, indent=2), encoding="utf-8"
)

print("Saved adapter to:", destination, flush=True)
print(json.dumps(metrics, indent=2))
