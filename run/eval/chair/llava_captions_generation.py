#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate single sentence captions for COCO val2014, output as:
[{"image_id": 391895, "caption": "..."} , ...]
Uses the same set of llava.* paths for loading and inference as in model_vqa_science.py.
"""

# -------- Important: Add the repository root to sys.path to ensure local llava code can be imported --------
import os, sys
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
# ---------------------------------------------------------------------------

import re, json, argparse
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from tqdm import tqdm

# —— Dependencies same as model_vqa_science.py —— #
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path


def collect_coco_val2014(img_dir):
    """Collect and return [(image_id, path), ...], used for pre-counting total number to display tqdm total."""
    pat = re.compile(r"COCO_val2014_0*([0-9]+)\.(jpg|jpeg|png)$", re.IGNORECASE)
    pairs = []
    for fname in sorted(os.listdir(img_dir)):
        m = pat.match(fname)
        if m:
            image_id = int(m.group(1))
            pairs.append((image_id, os.path.join(img_dir, fname)))
    return pairs


@torch.inference_mode()
def caption_one(model, tokenizer, image_processor, img_pil, prompt, device, conv_mode="llava_v1",
                max_new_tokens=64, temperature=0.0, top_p=None, num_beams=1):
    """
    Consistent with your existing pipeline: uses conv_templates + <image> + model.generate(images=...)
    """
    conv = conv_templates[conv_mode].copy()
    # Place image token at the start of the user message; same as in the ScienceQA script
    conv.append_message(conv.roles[0], f"{DEFAULT_IMAGE_TOKEN}\n{prompt}")
    conv.append_message(conv.roles[1], None)
    prompt_text = conv.get_prompt()

    # Visual preprocessing (returns [1,3,H,W])
    image_tensor = process_images([img_pil], image_processor, model.config)
    image_tensor = image_tensor.to(device=device, dtype=torch.float16)

    # Text tokens (map <image> location to IMAGE_TOKEN_INDEX)
    input_ids = tokenizer_image_token(
        prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)

    gen_kwargs = dict(
        input_ids=input_ids,
        images=image_tensor,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if temperature and temperature > 0:
        gen_kwargs.update(dict(do_sample=True, temperature=temperature, top_p=top_p if top_p is not None else 0.95))
    else:
        gen_kwargs.update(dict(do_sample=False))

    in_len = input_ids.shape[1]
    out = model.generate(**gen_kwargs)
    new_tokens = out[0, in_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to the merged LLaVA model")
    parser.add_argument("--model_base", default=None, help="Base model path, if applicable")
    parser.add_argument("--img_dir", required=True, help="Directory containing COCO val2014 images")
    parser.add_argument("--out_json", required=True, help="Path to output JSON file")
    parser.add_argument("--limit", type=int, default=-1, help="-1 for processing all images")
    parser.add_argument("--prompt", default="Describe this image.")
    parser.add_argument("--conv_mode", default="llava_v1", help="Use same conversation mode as existing scripts, default is llava_v1")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=0, help="Save every N captions; 0 means only save the last one")
    args = parser.parse_args()

    disable_torch_init()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model in the same way as your existing script
    model_name = get_model_name_from_path(os.path.expanduser(args.model_path))
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path, args.model_base, model_name, device_map="auto"
    )
    model.eval()

    out_path = os.path.expanduser(args.out_json)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Pre-collect file paths to get the total number
    pairs = collect_coco_val2014(args.img_dir)
    if args.limit and args.limit > 0:
        pairs = pairs[:args.limit]
    total = len(pairs)

    results = []
    fails = 0

    pbar = tqdm(
        pairs,
        total=total,
        desc="Captioning",
        dynamic_ncols=True,
        smoothing=0.1,
        mininterval=0.3,
        leave=True,
    )

    for image_id, img_path in pbar:
        try:
            img = Image.open(img_path).convert("RGB")
            cap = caption_one(
                model, tokenizer, image_processor, img, args.prompt, device,
                conv_mode=args.conv_mode,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature, top_p=args.top_p, num_beams=args.num_beams
            )
            results.append({"image_id": image_id, "caption": cap})
            # Progress bar suffix: display current image ID and first 30 characters of the caption
            pbar.set_postfix_str(f"id={image_id}, cap={cap[:30]!r}")
            # Save in batches (optional)
            if args.save_every and (len(results) % args.save_every == 0):
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
        except Exception as e:
            fails += 1
            pbar.set_postfix_str(f"id={image_id}, ERR={type(e).__name__}")
            continue

    # Final save
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(results)} captions to {out_path} | total={total}, fails={fails}")

if __name__ == "__main__":
    main()
