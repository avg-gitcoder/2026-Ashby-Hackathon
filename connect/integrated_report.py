"""One image -> PathGen caption -> MedGemma explanation + five MUSK predictions."""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

FEATURE_NAMES = {
    "abnormal_gland_arch": "Abnormal gland architecture",
    "nuclear_atypia": "Nuclear atypia",
    "abnormal_nuclear_org": "Abnormal nuclear organization",
    "stromal_invasion": "Stromal invasion",
    "intraluminal_necrosis": "Intraluminal necrosis",
}
WORDS = {-1: "no", 0: "unknown", 1: "yes"}


def render_report(record, prediction):
    text = record["simplified_caption"].strip() + "\n\n"
    labels = {}
    for feature, name in FEATURE_NAMES.items():
        value = int(prediction[feature])
        if value not in WORDS:
            raise ValueError(f"Invalid MUSK value for {feature}: {value}")
        labels[feature] = WORDS[value]
        text += f"{name}: {WORDS[value]}\n"
    return text, labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--musk-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--pathgen-model", required=True)
    parser.add_argument("--pathgen-python", default=sys.executable)
    parser.add_argument("--medgemma-python", default=sys.executable)
    parser.add_argument("--conv-mode", default="llava_v1")
    parser.add_argument("--medgemma-base", default="google/medgemma-4b-it")
    args = parser.parse_args()
    image = args.image.resolve(strict=True)
    checkpoint = args.musk_checkpoint.resolve(strict=True)
    # Fresh run directory prevents stale files from a different image being merged.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir.resolve()
    scripts = Path(__file__).resolve().parent
    digest = hashlib.sha256(image.read_bytes()).hexdigest()

    def run(python, script, *values):
        print(f"Running {script}", flush=True)
        subprocess.run([python, str(scripts/script), *map(str, values)], check=True)
        if hashlib.sha256(image.read_bytes()).hexdigest() != digest:
            raise RuntimeError("Input image changed; refusing to merge results.")

    # Each subprocess exits before the next model loads, releasing its GPU memory.
    run(args.pathgen_python, "caption_pathgen.py", "--image", image,
        "--model", args.pathgen_model, "--conv-mode", args.conv_mode,
        "--output", output/"caption.json")
    run(args.medgemma_python, "simplify_medgemma.py", "--caption-json", output/"caption.json",
        "--base-model", args.medgemma_base, "--output", output/"explanation.json")
    run(sys.executable, scripts.parent / "musk" / "infer_morphology.py", "--image", image,
        "--checkpoint", checkpoint, "--output", output/"musk.csv")
    record = json.loads((output/"explanation.json").read_text(encoding="utf-8"))
    if record["image_sha256"] != digest or Path(record["image_path"]).resolve() != image:
        raise ValueError("Caption image identity mismatch")
    with (output/"musk.csv").open(encoding="utf-8", newline="") as stream:
        predictions = list(csv.DictReader(stream))
    if len(predictions) != 1 or Path(predictions[0]["image_path"]).resolve() != image:
        raise ValueError("MUSK image identity mismatch")
    text, labels = render_report(record, predictions[0])
    record.update(morphology=labels, musk_prediction=predictions[0],
                  musk_checkpoint=str(checkpoint), report=text)
    (output/"report.txt").write_text(text, encoding="utf-8")
    (output/"report.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(text, flush=True)
    print(f"Saved report: {output/'report.txt'}", flush=True)


if __name__ == "__main__":
    main()
