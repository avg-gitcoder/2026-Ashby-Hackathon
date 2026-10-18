"""Batch-safe caption simplification using the repository's existing adapter."""
import argparse
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--caption-json", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--adapter", type=Path, default=root / "medgem/medgemma_simplifier_adapter")
    p.add_argument("--instruction", type=Path, default=root / "medgem/fewshot_instruction.txt")
    p.add_argument("--base-model", default="google/medgemma-4b-it")
    args = p.parse_args()
    if args.output.exists():
        p.error("Output already exists")
    record = json.loads(args.caption_json.read_text(encoding="utf-8"))
    if not isinstance(record.get("caption"), str) or not record["caption"].strip():
        raise ValueError("Missing caption")
    import torch
    from transformers import AutoTokenizer, Gemma3ForConditionalGeneration
    from peft import PeftModel

    if not torch.cuda.is_available():
        raise RuntimeError("Run MedGemma in a GPU Slurm job.")
    tokenizer = AutoTokenizer.from_pretrained(args.adapter, local_files_only=True)
    base = Gemma3ForConditionalGeneration.from_pretrained(args.base_model,
        local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
    model = PeftModel.from_pretrained(base, args.adapter, local_files_only=True).eval()
    instruction = args.instruction.read_text(encoding="utf-8")
    messages = [{"role": "user", "content": instruction + "\n\nCaption:\n" + record["caption"]}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=768, do_sample=False,
                                use_cache=True, pad_token_id=tokenizer.pad_token_id)
    answer = tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    if not answer:
        raise RuntimeError("MedGemma generated an empty explanation")
    record.update(simplified_caption=answer, medgemma_base=args.base_model,
                  medgemma_adapter=str(args.adapter.resolve()))
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
