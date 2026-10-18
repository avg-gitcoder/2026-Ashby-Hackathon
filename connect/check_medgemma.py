"""Check MedGemma files/dependencies without loading model weights or using a GPU."""
import argparse
import json
from pathlib import Path

LOCAL_BASE = Path("/projects/bfqi/jkim75/models/medgemma-4b-it")


def default_base_model():
    return str(LOCAL_BASE) if (LOCAL_BASE / "config.json").is_file() else "google/medgemma-4b-it"


def check_weight_files(folder):
    folder = Path(folder)
    if not (folder / "config.json").is_file():
        raise FileNotFoundError(f"Missing model config: {folder}")
    index = folder / "model.safetensors.index.json"
    if index.is_file():
        names = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
        if not names:
            raise ValueError("Empty model shard index")
    else:
        names = {"model.safetensors"}
    for name in names:
        file = (folder / name).resolve()
        if not file.is_relative_to(folder.resolve()):
            # HF cache snapshots use symlinks into the blobs directory; permit those
            # provided the index itself is a plain basename without path traversal.
            if Path(name).name != name:
                raise ValueError(f"Invalid shard name: {name}")
        if not file.is_file() or file.stat().st_size < 1024 * 1024:
            raise FileNotFoundError(f"Missing/incomplete model weight: {folder / name}")


def check(base_model, adapter):
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import AutoTokenizer, AutoConfig, Gemma3ForConditionalGeneration

    folder = Path(base_model)
    if not folder.is_dir():
        try:
            folder = Path(snapshot_download(base_model, local_files_only=True))
        except Exception as error:
            raise RuntimeError("MedGemma base weights are not available in this environment. "
                "From connect/, run: python check_medgemma.py --download --model-dir "
                + str(LOCAL_BASE)) from error
    check_weight_files(folder)
    AutoConfig.from_pretrained(str(folder), local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(adapter, local_files_only=True)
    if not tokenizer.chat_template:
        raise ValueError("MedGemma adapter tokenizer is missing a chat template")
    adapter = Path(adapter)
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        if not (adapter / name).is_file():
            raise FileNotFoundError(adapter / name)
    return folder


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", default=default_base_model())
    p.add_argument("--adapter", type=Path,
                   default=Path(__file__).resolve().parent.parent / "medgem/medgemma_simplifier_adapter")
    p.add_argument("--download", action="store_true")
    p.add_argument("--model-dir", type=Path, default=LOCAL_BASE)
    args = p.parse_args()
    base = args.base_model
    if args.download:
        from huggingface_hub import snapshot_download
        snapshot_download("google/medgemma-4b-it", local_dir=str(args.model_dir),
            allow_patterns=["*.json", "*.safetensors", "*.model", "*.jinja", "*.txt"])
        base = str(args.model_dir)
    folder = check(base, args.adapter)
    print(f"MedGemma preflight OK: {folder}")


if __name__ == "__main__":
    main()
