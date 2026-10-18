"""Download official PathGen-LLaVA and configure its local vision-tower path."""
import argparse
import json
from pathlib import Path


def configure_model(model_dir):
    model_dir = Path(model_dir).resolve()
    config_file = model_dir / "config.json"
    tower = model_dir / "pathgenclip-vit-large-patch14"
    if not (tower / "config.json").is_file():
        raise FileNotFoundError(f"Vision config not found: {tower / 'config.json'}")
    weights = [*tower.glob("*.bin"), *tower.glob("*.safetensors")]
    if not weights or not any(p.stat().st_size > 1024 * 1024 for p in weights):
        raise ValueError(f"Vision weights missing or appear to be Git LFS pointers: {tower}")
    config = json.loads(config_file.read_text(encoding="utf-8"))
    fields = [k for k in ("mm_vision_tower", "vision_tower") if k in config]
    if not fields:
        raise ValueError("No vision tower field found in model config; inspect the checkpoint format.")
    backup = config_file.with_name("config.original.json")
    if not backup.exists():
        backup.write_bytes(config_file.read_bytes())
    for field in fields:
        config[field] = str(tower)
    config_file.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return tower


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", required=True, type=Path)
    p.add_argument("--revision", default="main")
    args = p.parse_args()
    from huggingface_hub import HfApi, snapshot_download

    # Resolve once to an immutable revision; preserve it for reproducibility.
    revision = HfApi().model_info("jamessyx/PathGen-LLaVA", revision=args.revision).sha
    snapshot_download("jamessyx/PathGen-LLaVA", revision=revision, local_dir=str(args.model_dir))
    tower = configure_model(args.model_dir)
    (args.model_dir / "download_revision.txt").write_text(revision + "\n", encoding="utf-8")
    print(f"Downloaded: {args.model_dir.resolve()}")
    print(f"Vision tower: {tower}")
    print(f"Revision: {revision}")
    print("Download/config complete. GPU inference must be tested in a Slurm job.")


if __name__ == "__main__":
    main()
