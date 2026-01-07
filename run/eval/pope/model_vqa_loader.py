import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import argparse
import json
import math
from tqdm import tqdm
import shortuuid
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from functools import partial

def get_num_image_tokens(image_tensor, model):
    """
    Input: image_tensor [B,3,H,W] (fp32/fp16 both OK)
    Returns: token length after vision tower processing (based on first image)
    """
    with torch.no_grad():
        if hasattr(model, "encode_images"):
            t = image_tensor
            if t.dim() == 3:
                t = t.unsqueeze(0)  # [1,3,H,W]
            t = t.to(device="cuda", dtype=torch.float16)
            img_embeds = model.encode_images(t)  # [B, T_img, hidden]
            return int(img_embeds.shape[1])
        # Fallback: estimate from vision_tower
        vt = model.get_vision_tower() if hasattr(model, "get_vision_tower") else None
        if vt is None:
            return 0
        t = image_tensor
        if t.dim() == 3:
            t = t.unsqueeze(0)
        t = t.to(device="cuda", dtype=next(vt.parameters()).dtype)
        out = vt(t)
        if hasattr(out, "last_hidden_state"):
            return int(out.last_hidden_state.shape[1])
        elif isinstance(out, (tuple, list)):
            return int(out[0].shape[1])
        else:
            return int(out.shape[1])


def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    return split_list(lst, n)[k]


class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config, conv_mode):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.conv_mode = conv_mode

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]

        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        img = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        img_tensor = process_images([img], self.image_processor, self.model_config)[0]  # [3,H,W]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')  # [1, T]

        return input_ids.squeeze(0).long(), img_tensor, img.size  # [T], [3,H,W], (W,H)

    def __len__(self):
        return len(self.questions)


def collate_fn(batch, pad_id=0):
    input_ids_list, img_tensors_list, image_sizes = zip(*batch)
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id)
    images = torch.stack(img_tensors_list, dim=0)
    return input_ids, images, image_sizes


def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, conv_mode, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config, conv_mode)
    return DataLoader(dataset, batch_size=1, num_workers=num_workers, shuffle=False,
                      collate_fn=partial(collate_fn, pad_id=tokenizer.pad_token_id or 0))

def eval_model(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    model.eval()

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    # === Register LLaVA special tokens and align ===
    from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

    added = tokenizer.add_special_tokens({
        "additional_special_tokens": [DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN]
    })

    # pad / eos alignment (common for LLaMA)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # Resize if vocab size changed or model embeddings don't match tokenizer
    if added > 0 or model.get_input_embeddings().weight.shape[0] != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
        if hasattr(model, "tie_weights"):
            model.tie_weights()

    print(f"[INFO] tokenizer size={len(tokenizer)}, embed size={model.get_input_embeddings().weight.shape[0]}")
    # === End ===

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config, args.conv_mode)

    if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
        if hasattr(model, "tie_weights"):
            model.tie_weights()

    for (input_ids, images, image_sizes), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["question_id"]
        cur_prompt = line["text"]

        # Device/precision
        input_ids = input_ids.to(device='cuda', non_blocking=True)              # [B,T]
        images = images.to(device='cuda', dtype=torch.float16, non_blocking=True)  # [B,3,H,W]

        # ====== Calculate image tokens and apply RoPE safety margin ======
        max_pos = getattr(model.config, "max_position_embeddings", None)
        num_img_placeholders = int((input_ids == IMAGE_TOKEN_INDEX).sum().item())
        num_img_tokens = get_num_image_tokens(images, model) if num_img_placeholders > 0 else 0

        text_len = input_ids.size(1)
        effective_prefill_len = text_len - num_img_placeholders + num_img_placeholders * num_img_tokens

        if max_pos is not None:
            max_prefill = max(1, max_pos - 1)
            if effective_prefill_len > max_prefill:
                need_trim = effective_prefill_len - max_prefill
                keep = max(1, text_len - need_trim)
                input_ids = input_ids[:, -keep:]
                text_len = input_ids.size(1)
                num_img_placeholders = int((input_ids == IMAGE_TOKEN_INDEX).sum().item())
                effective_prefill_len = text_len - num_img_placeholders + num_img_placeholders * num_img_tokens
                if effective_prefill_len > max_prefill:
                    input_ids = input_ids[:, -1:]
                    text_len = 1
                    num_img_placeholders = int((input_ids == IMAGE_TOKEN_INDEX).sum().item())
                    effective_prefill_len = text_len - num_img_placeholders + num_img_placeholders * num_img_tokens

            allowed_new = max(0, max_pos - effective_prefill_len)
            safe_max_new = min(args.max_new_tokens, allowed_new)
            # Additional check from pure text perspective
            safe_max_new = min(safe_max_new, max(0, int(max_pos or 4096) - input_ids.size(1) - 1))
        else:
            safe_max_new = args.max_new_tokens

        # Debug prints
        # if images.dim() == 4:
        #     print(f"img.shape={tuple(images.shape)}")
        # print(f"num_img_tokens={num_img_tokens}")
        # print(f"max_pos={max_pos}, prefill_len={effective_prefill_len}, safe_max_new={safe_max_new}")
        # print(f"image_sizes={image_sizes}")

        if safe_max_new <= 0:
            outputs = ""
            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({
                "question_id": idx,
                "prompt": cur_prompt,
                "text": outputs,
                "answer_id": ans_id,
                "model_id": model_name,
                "metadata": {
                    "note": "truncated context due to max_position_embeddings",
                    "max_pos": int(max_pos) if max_pos is not None else None,
                    "num_img_tokens": int(num_img_tokens),
                    "prefill_len": int(effective_prefill_len)
                }
            }) + "\n")
            continue

        # ====== Generation: only pass pixel tensor, no lists or image_sizes ======
        with torch.inference_mode():
            do_sample = (args.temperature is not None and args.temperature > 0)
            gen_kwargs = dict(
                input_ids=input_ids,
                images=images,
                num_beams=args.num_beams,
                max_new_tokens=safe_max_new,
                use_cache=True
            )
            if do_sample:
                gen_kwargs.update(dict(
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p if args.top_p is not None else 0.95
                ))

            input_len = input_ids.size(1)
            output_ids = model.generate(**gen_kwargs)

        # Decode only newly generated tokens
        gen_only = output_ids[:, input_len:]
        outputs = tokenizer.batch_decode(
            gen_only, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0].strip()

        # (Optional debug)
        # print("***"); print(output_ids); print(tokenizer.vocab_size)

        # Write results
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({
            "question_id": idx,
            "prompt": cur_prompt,
            "text": outputs,
            "answer_id": ans_id,
            "model_id": model_name,
            "metadata": {}
        }) + "\n")

    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new-tokens", type=int, default=128)
    args = parser.parse_args()

    eval_model(args)