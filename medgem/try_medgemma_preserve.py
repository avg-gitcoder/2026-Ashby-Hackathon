from pathlib import Path
import torch
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration
from peft import PeftModel

folder = Path(__file__).resolve().parent
adapter_path = folder / "medgemma_simplifier_adapter"
model_id = "google/medgemma-4b-it"

assert torch.cuda.is_available(), "Run this on your GPU node."

tokenizer = AutoTokenizer.from_pretrained(
    adapter_path, local_files_only=True
)

base_model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id,
    local_files_only=True,
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
).to("cuda")

model = PeftModel.from_pretrained(
    base_model, adapter_path, local_files_only=True
)
model.eval()
model.config.use_cache = True
model.config.text_config.use_cache = True

instruction = (folder / "preservation_instruction.txt").read_text(
    encoding="utf-8"
)

print("\nModel ready. Enter a caption on one line.")
print("Type 'quit' to exit.\n")

while True:
    caption = input("Medical caption: ").strip()

    if caption.lower() == "quit":
        break
    if not caption:
        continue

    messages = [{
        "role": "user",
        "content": instruction + "\n\nCaption:\n" + caption,
    }]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(
        prompt, add_special_tokens=False, return_tensors="pt"
    ).to("cuda")

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    answer = tokenizer.decode(
        output[0, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    print("\n=== Simplified caption ===")
    print(answer)
    print()
