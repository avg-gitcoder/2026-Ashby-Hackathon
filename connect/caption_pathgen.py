"""Single-image PathGen-LLaVA captioning; run inside the separate LLaVA environment."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", required=True, type=Path)
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--conv-mode", default="llava_v1")
    args = p.parse_args()
    if args.output.exists():
        p.error("Output already exists")
    digest = hashlib.sha256(args.image.read_bytes()).hexdigest()
    import torch
    from PIL import Image
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from llava.conversation import conv_templates
    from llava.mm_utils import process_images, tokenizer_image_token
    from llava.model.builder import load_pretrained_model

    if not torch.cuda.is_available():
        raise RuntimeError("Run captioning in a GPU Slurm job.")
    # Explicit llava name: a local folder named PathGen-LLaVA must use LLaVA's loader.
    tokenizer, model, processor, _ = load_pretrained_model(args.model, None, "llava-v1.5-pathgen")
    model.eval()
    image_token = DEFAULT_IMAGE_TOKEN
    if model.config.mm_use_im_start_end:
        image_token = DEFAULT_IM_START_TOKEN + image_token + DEFAULT_IM_END_TOKEN
    conv = conv_templates[args.conv_mode].copy()
    query = "Describe the histopathological findings visible in this image. State uncertainty and assessment limitations; do not invent findings that are not visible."
    conv.append_message(conv.roles[0], image_token + "\n" + query)
    conv.append_message(conv.roles[1], None)
    ids = tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX,
                                return_tensors="pt").unsqueeze(0).to(model.device)
    with Image.open(args.image) as source:
        image = source.convert("RGB")
    pixels = process_images([image], processor, model.config)
    if isinstance(pixels, list):
        pixels = [x.to(model.device, dtype=torch.float16) for x in pixels]
    else:
        pixels = pixels.to(model.device, dtype=torch.float16)
    with torch.inference_mode():
        output = model.generate(ids, images=pixels, image_sizes=[image.size],
                                do_sample=False, max_new_tokens=512, use_cache=True)
    caption = tokenizer.batch_decode(output, skip_special_tokens=True)[0].strip()
    if not caption:
        raise RuntimeError("PathGen generated an empty caption.")
    if hashlib.sha256(args.image.read_bytes()).hexdigest() != digest:
        raise RuntimeError("Image changed during caption generation.")
    record = {"image_path": str(args.image.resolve()), "image_sha256": digest,
              "caption": caption, "caption_model": args.model, "conv_mode": args.conv_mode}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
