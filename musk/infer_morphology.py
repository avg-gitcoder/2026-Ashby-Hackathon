"""Inference using the full best.pt saved by train_morphology.py."""
import argparse
import csv
from pathlib import Path

FEATURES = (
    "abnormal_gland_arch", "nuclear_atypia", "abnormal_nuclear_org",
    "stromal_invasion", "intraluminal_necrosis",
)
LABEL_MAPPING = {"no": -1, "unknown": 0, "yes": 1}
DEFAULT_IMAGES = Path("/projects/bfqi/shared/datasets/NCT-CRC-HE-100K/data-samples/TUM")


def collect_images(args):
    if args.image:
        paths = [args.image]
    elif args.csv:
        with args.csv.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if "filename" not in (reader.fieldnames or []):
                raise ValueError("CSV must include a filename column.")
            names = [row["filename"] for row in reader]
        for name in names:
            if not name or Path(name).name != name or "\\" in name:
                raise ValueError(f"Expected a filename without directory components: {name!r}")
        if len(set(names)) != len(names):
            raise ValueError("Duplicate filenames in CSV.")
        paths = [args.image_root / name for name in names]
    else:
        paths = sorted(p for p in args.input_dir.iterdir()
                       if p.is_file() and p.suffix.lower() in {".tif", ".tiff", ".png", ".jpg", ".jpeg"})
    if not paths:
        raise ValueError("No images selected.")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="One image")
    source.add_argument("--csv", type=Path, help="CSV containing filenames, e.g. val.csv")
    source.add_argument("--input-dir", type=Path, help="All supported images in one directory")
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")
    if args.output.exists():
        parser.error("Output already exists; choose a new filename.")
    paths = collect_images(args)

    import torch
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode
    from PIL import Image
    from musk import modeling  # register the model
    from timm.models import create_model
    from morphology_head import MorphologyModel

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Submit a GPU job; do not run inference on a login node.")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if saved.get("features") != list(FEATURES) or saved.get("label_mapping") != LABEL_MAPPING:
        raise ValueError("Expected a full training checkpoint with matching feature and label definitions.")
    # best.pt already contains the encoder. No HF login or new download is required.
    model = MorphologyModel(create_model("musk_large_patch16_384"))
    model.load_state_dict(saved["model"], strict=True)
    del saved
    model.to(args.device).eval()
    transform = transforms.Compose([
        transforms.Resize((384, 384), interpolation=InterpolationMode.BICUBIC, antialias=True),
        transforms.ToTensor(), transforms.Normalize((.5, .5, .5), (.5, .5, .5)),
    ])
    columns = ["filename", "image_path"]
    for feature in FEATURES:
        columns += [feature, f"{feature}_confidence", f"{feature}_p_no",
                    f"{feature}_p_unknown", f"{feature}_p_yes"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # A failed run leaves a partial file; the final CSV is published only on success.
    partial = args.output.with_name(args.output.name + ".partial")
    with partial.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for start in range(0, len(paths), args.batch_size):
            batch_paths = paths[start:start + args.batch_size]
            tensors = []
            for path in batch_paths:
                with Image.open(path) as image:
                    tensors.append(transform(image.convert("RGB")))
            # Match validation's CUDA mixed precision, with eval mode and no random augmentation.
            with torch.inference_mode(), torch.cuda.amp.autocast(enabled=args.device == "cuda"):
                logits = model(torch.stack(tensors).to(args.device))
            probabilities = logits.float().softmax(-1).cpu()
            confidence, indices = probabilities.max(-1)
            for i, path in enumerate(batch_paths):
                row = {"filename": path.name, "image_path": str(path)}
                for j, feature in enumerate(FEATURES):
                    row[feature] = int(indices[i, j]) - 1
                    row[f"{feature}_confidence"] = float(confidence[i, j])
                    for k, label in enumerate(("no", "unknown", "yes")):
                        row[f"{feature}_p_{label}"] = float(probabilities[i, j, k])
                writer.writerow(row)
                if len(paths) == 1:
                    print({feature: row[feature] for feature in FEATURES}, flush=True)
            stream.flush()
            print(f"Predicted {start + len(batch_paths)}/{len(paths)}", flush=True)
    partial.rename(args.output)
    print(f"Saved predictions: {args.output}", flush=True)


if __name__ == "__main__":
    main()
