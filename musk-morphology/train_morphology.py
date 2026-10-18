"""CSV-driven MUSK training. Run --check-data before training."""
import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

FEATURES = (
    "abnormal_gland_arch", "nuclear_atypia", "abnormal_nuclear_org",
    "stromal_invasion", "intraluminal_necrosis",
)
LABELS = {"no": -1, "unknown": 0, "yes": 1}


def read_rows(csv_path, image_root, group_column=None):
    with open(csv_path, encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"filename", *FEATURES}
        if group_column:
            required.add(group_column)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing columns: {required - set(reader.fieldnames or [])}")
        rows = list(reader)
    if len(rows) < 3:
        raise ValueError("At least 3 images are required.")
    seen = set()
    for row in rows:
        filename = row["filename"]
        if not filename or Path(filename).name != filename or "\\" in filename:
            raise ValueError(f"Expected a basename: {filename!r}")
        if filename in seen:
            raise ValueError(f"Duplicate filename: {filename}")
        seen.add(filename)
        for feature in FEATURES:
            row[feature] = row[feature].strip().lower()
            if row[feature] not in LABELS:
                raise ValueError(f"Invalid label in {filename}, {feature}: {row[feature]!r}")
        if group_column and not row[group_column].strip():
            raise ValueError(f"Empty group in {filename}")
        if not (image_root / filename).is_file():
            raise FileNotFoundError(image_root / filename)
    return rows


def counts(rows):
    return {f: {label: sum(r[f] == label for r in rows) for label in LABELS} for f in FEATURES}


def split_rows(rows, fraction, seed, group_column=None):
    rng = random.Random(seed)
    if group_column:
        groups = sorted({r[group_column] for r in rows})
        if len(groups) < 2:
            raise ValueError("At least 2 patient/slide groups are required.")
        rng.shuffle(groups)
        val_groups = set(groups[:max(1, min(len(groups)-1, round(len(groups)*fraction)))])
        return ([r for r in rows if r[group_column] not in val_groups],
                [r for r in rows if r[group_column] in val_groups])
    shuffled = list(rows)
    rng.shuffle(shuffled)
    size = max(1, min(len(rows)-1, round(len(rows)*fraction)))
    return shuffled[size:], shuffled[:size]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--image-root", type=Path,
                        default=Path("/projects/bfqi/shared/datasets/NCT-CRC-HE-100K/data-samples/TUM"))
    parser.add_argument("--output", type=Path, default=Path("runs/morphology"))
    parser.add_argument("--check-data", action="store_true")
    parser.add_argument("--group-column", help="Patient/slide ID; never use image-level index")
    parser.add_argument("--allow-patch-split", action="store_true",
                        help="Exploratory only: patient overlap cannot be excluded")
    parser.add_argument("--val-fraction", type=float, default=.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--finetune-encoder", action="store_true")
    parser.add_argument("--init", type=Path, help="Our trained checkpoint to initialize from")
    parser.add_argument("--pretrained", default="hf_hub:xiangjx/musk",
                        help="Official safetensors path or HF source")
    parser.add_argument("--class-weighting", choices=["none", "sqrt-inverse"], default="none")
    args = parser.parse_args()
    if not 0 < args.val_fraction < 1 or min(args.epochs, args.batch_size) < 1 or args.workers < 0:
        parser.error("Invalid fraction, epochs, batch size, or workers")
    rows = read_rows(args.csv, args.image_root, args.group_column)
    report = {"images": len(rows), "labels": counts(rows)}
    print(json.dumps(report, indent=2))
    for feature, distribution in report["labels"].items():
        missing = [label for label, n in distribution.items() if n == 0]
        if missing:
            print(f"WARNING: {feature}: no examples of {missing}; these classes cannot be learned.")
    if args.check_data:
        return
    if not args.group_column and not args.allow_patch_split:
        parser.error("Provide --group-column or explicitly use --allow-patch-split for a pilot.")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory must be empty; choose a new --output.")
    train_rows, val_rows = split_rows(rows, args.val_fraction, args.seed, args.group_column)

    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode
    from PIL import Image
    from morphology_head import load_pretrained

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)
    for name, subset in (("train", train_rows), ("val", val_rows)):
        with (args.output / f"{name}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(subset)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(features=list(FEATURES), label_mapping=LABELS,
                  train_counts=counts(train_rows), val_counts=counts(val_rows),
                  split_warning=None if args.group_column else "Patch split: possible patient/slide leakage")
    (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    class Patches(Dataset):
        def __init__(self, records, training):
            self.records = records
            ops = [transforms.Resize((384, 384), interpolation=InterpolationMode.BICUBIC, antialias=True)]
            if training:
                ops += [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
            self.transform = transforms.Compose(ops + [transforms.ToTensor(),
                transforms.Normalize((.5, .5, .5), (.5, .5, .5))])

        def __len__(self):
            return len(self.records)

        def __getitem__(self, index):
            row = self.records[index]
            with Image.open(args.image_root / row["filename"]) as image:
                tensor = self.transform(image.convert("RGB"))
            return tensor, torch.tensor([LABELS[row[f]] + 1 for f in FEATURES])

    # Linux server loader; workers=0 is also supported for debugging.
    train_loader = DataLoader(Patches(train_rows, True), batch_size=args.batch_size,
        shuffle=True, num_workers=args.workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(Patches(val_rows, False), batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=device.type == "cuda")
    model = load_pretrained(not args.finetune_encoder, args.pretrained).to(device)
    if args.init:
        saved = torch.load(args.init, map_location="cpu", weights_only=True)
        if saved["features"] != list(FEATURES) or saved["label_mapping"] != LABELS:
            raise ValueError("Checkpoint label schema differs.")
        model.load_state_dict(saved["model"])
    params = [{"params": model.head.parameters(), "lr": args.head_lr}]
    if args.finetune_encoder:
        params.append({"params": model.encoder.parameters(), "lr": args.encoder_lr})
    optimizer = torch.optim.AdamW(params, weight_decay=.01)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    weights = torch.ones(5, 3, device=device)
    if args.class_weighting == "sqrt-inverse":
        for j, feature in enumerate(FEATURES):
            n = torch.tensor([sum(r[feature] == label for r in train_rows)
                              for label in LABELS], device=device, dtype=torch.float)
            weights[j] = torch.where(n > 0, n.clamp_min(1).rsqrt(), 0)
            weights[j] /= weights[j].sum() / (n > 0).sum()

    def loss_fn(logits, targets):
        return torch.stack([F.cross_entropy(logits[:, j].float(), targets[:, j],
                           weight=weights[j]) for j in range(5)]).mean()

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                loss = loss_fn(model(images), targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            scaler.step(optimizer)
            scaler.update()
            total += loss.item() * len(images)
        model.eval()
        val_loss = 0.
        confusion = torch.zeros(5, 3, 3, dtype=torch.long)
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                    logits = model(images)
                    # Unweighted validation loss remains defined for classes missing in train.
                    loss = F.cross_entropy(logits.float().reshape(-1, 3), targets.reshape(-1))
                val_loss += loss.item() * len(images)
                predicted = logits.argmax(-1)
                for j in range(5):
                    confusion[j] += torch.bincount((targets[:, j]*3 + predicted[:, j]).cpu(),
                                                   minlength=9).reshape(3, 3)
        metrics = {}
        for j, feature in enumerate(FEATURES):
            matrix = confusion[j].float()
            denominator = matrix.sum(0) + matrix.sum(1)
            f1 = 2 * matrix.diag() / denominator.clamp_min(1)
            metrics[feature] = {"macro_f1_fixed_3_classes": f1.mean().item(),
                "accuracy": (matrix.diag().sum()/matrix.sum()).item(),
                "support_no_unknown_yes": matrix.sum(1).long().tolist(),
                "confusion_true_rows_pred_columns": confusion[j].tolist()}
        record = {"epoch": epoch, "train_loss": total/len(train_rows),
                  "val_loss": val_loss/len(val_rows), "features": metrics}
        print(json.dumps(record), flush=True)
        with (args.output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        if record["val_loss"] < best_loss:
            best_loss = record["val_loss"]
            torch.save({"model": model.state_dict(), "features": list(FEATURES),
                "label_mapping": LABELS, "config": config, "epoch": epoch,
                "val_loss": best_loss}, args.output / "best.pt")


if __name__ == "__main__":
    main()
