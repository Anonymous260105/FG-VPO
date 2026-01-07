import argparse
import torch
import os
import json
import pandas as pd
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, load_image_from_base64, get_model_name_from_path

from PIL import Image
import math

def get_num_image_tokens(image_tensor, model):
    """
    Input: image_tensor [B,3,H,W] (fp32/fp16 are both acceptable)
    Output: The number of tokens corresponding to the batch of images after being processed by the vision tower (based on the first image)
    """
    with torch.no_grad():
        if hasattr(model, "encode_images"):
            t = image_tensor
            if t.dim() == 3:
                t = t.unsqueeze(0)  # [1,3,H,W]
            t = t.to(device="cuda", dtype=torch.float16)
            img_embeds = model.encode_images(t)  # [B, T_img, hidden]
            return int(img_embeds.shape[1])
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


all_options = ['A', 'B', 'C', 'D']


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def is_none(value):
    if value is None:
        return True
    if type(value) is float and math.isnan(value):
        return True
    if type(value) is str and value.lower() == 'nan':
        return True
    if type(value) is str and value.lower() == 'none':
        return True
    return False

def get_options(row, options):
    parsed_options = []
    for option in options:
        option_value = row[option]
        if is_none(option_value):
            break
        parsed_options.append(option_value)
    return parsed_options

def eval_model(args):
    # Model initialization
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # === Register LLaVA special tokens and alignment (consistent with reference) ===
    added = tokenizer.add_special_tokens({
        "additional_special_tokens": [DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN]
    })
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if added > 0 or model.get_input_embeddings().weight.shape[0] != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
        if hasattr(model, "tie_weights"):
            model.tie_weights()
    # === End ===

    questions = pd.read_table(os.path.expanduser(args.question_file))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    for index, row in tqdm(questions.iterrows(), total=len(questions)):
        options = get_options(row, all_options)
        cur_option_char = all_options[:len(options)]

        if args.all_rounds:
            num_rounds = len(options)
        else:
            num_rounds = 1

        for round_idx in range(num_rounds):
            idx = row['index']
            question = row['question']
            hint = row['hint']
            image = load_image_from_base64(row['image'])
            if not is_none(hint):
                question = hint + '\n' + question
            for option_char, option in zip(all_options[:len(options)], options):
                question = question + '\n' + option_char + '. ' + option
            qs = cur_prompt = question
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            if args.single_pred_prompt:
                if args.lang == 'cn':
                    qs = qs + '\n' + "请直接回答选项字母。"
                else:
                    qs = qs + '\n' + "Answer with the option's letter from the given choices directly."

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            # Ensure input_ids is [1, T]
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
            if isinstance(input_ids, list):
                input_ids = torch.tensor(input_ids, dtype=torch.long)  # For compatibility with edge cases
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)  # [1, T]

            # Move to model's device
            param = next(model.parameters())
            input_ids = input_ids.to(device=param.device)

            img_tensor = process_images([image], image_processor, model.config)[0]      # [3,H,W]
            images = img_tensor.unsqueeze(0).to(device=param.device, dtype=param.dtype)  # [1,3,H,W]

            # ====== Calculate effective prefill length and trim max_new_tokens (before generate)======
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
                    input_ids = input_ids[:, -keep:]  # Trim from the right to keep the end
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
                # Make it conservative again: ensure position limit - existing token - at least 1 token
                safe_max_new = min(safe_max_new, max(0, int(max_pos or 4096) - input_ids.size(1) - 1))
            else:
                safe_max_new = args.max_new_tokens

            # ====== Actual generation (no image_sizes passed; decode only new tokens)======
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

            # Decode only the newly generated tokens, avoid decoding the prompt
            gen_only = output_ids[:, input_len:]
            outputs = tokenizer.batch_decode(gen_only, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0].strip()

            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({"question_id": idx,
                                    "round_id": round_idx,
                                    "prompt": cur_prompt,
                                    "text": outputs,
                                    "options": options,
                                    "option_char": cur_option_char,
                                    "answer_id": ans_id,
                                    "model_id": model_name,
                                    "metadata": {}}) + "\n")
            ans_file.flush()

            # Rotate options
            options = options[1:] + options[:1]
            cur_option_char = cur_option_char[1:] + cur_option_char[:1]
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
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--all-rounds", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--max-new-tokens", type=int, default=128)

    args = parser.parse_args()

    eval_model(args)
import argparse
import torch
import os
import json
import pandas as pd
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, load_image_from_base64, get_model_name_from_path

from PIL import Image
import math

def get_num_image_tokens(image_tensor, model):
    """
    Input: image_tensor [B,3,H,W] (fp32/fp16 are both acceptable)
    Output: The number of tokens corresponding to the batch of images after being processed by the vision tower (based on the first image)
    """
    with torch.no_grad():
        if hasattr(model, "encode_images"):
            t = image_tensor
            if t.dim() == 3:
                t = t.unsqueeze(0)  # [1,3,H,W]
            t = t.to(device="cuda", dtype=torch.float16)
            img_embeds = model.encode_images(t)  # [B, T_img, hidden]
            return int(img_embeds.shape[1])
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


all_options = ['A', 'B', 'C', 'D']


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def is_none(value):
    if value is None:
        return True
    if type(value) is float and math.isnan(value):
        return True
    if type(value) is str and value.lower() == 'nan':
        return True
    if type(value) is str and value.lower() == 'none':
        return True
    return False

def get_options(row, options):
    parsed_options = []
    for option in options:
        option_value = row[option]
        if is_none(option_value):
            break
        parsed_options.append(option_value)
    return parsed_options

def eval_model(args):
    # Model initialization
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # === Register LLaVA special tokens and alignment (consistent with reference) ===
    added = tokenizer.add_special_tokens({
        "additional_special_tokens": [DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN]
    })
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if added > 0 or model.get_input_embeddings().weight.shape[0] != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
        if hasattr(model, "tie_weights"):
            model.tie_weights()
    # === End ===

    questions = pd.read_table(os.path.expanduser(args.question_file))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    for index, row in tqdm(questions.iterrows(), total=len(questions)):
        options = get_options(row, all_options)
        cur_option_char = all_options[:len(options)]

        if args.all_rounds:
            num_rounds = len(options)
        else:
            num_rounds = 1

        for round_idx in range(num_rounds):
            idx = row['index']
            question = row['question']
            hint = row['hint']
            image = load_image_from_base64(row['image'])
            if not is_none(hint):
                question = hint + '\n' + question
            for option_char, option in zip(all_options[:len(options)], options):
                question = question + '\n' + option_char + '. ' + option
            qs = cur_prompt = question
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            if args.single_pred_prompt:
                if args.lang == 'cn':
                    qs = qs + '\n' + "请直接回答选项字母。"
                else:
                    qs = qs + '\n' + "Answer with the option's letter from the given choices directly."

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            # Ensure input_ids is [1, T]
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
            if isinstance(input_ids, list):
                input_ids = torch.tensor(input_ids, dtype=torch.long)  # For compatibility with edge cases
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)  # [1, T]

            # Move to model's device
            param = next(model.parameters())
            input_ids = input_ids.to(device=param.device)

            img_tensor = process_images([image], image_processor, model.config)[0]      # [3,H,W]
            images = img_tensor.unsqueeze(0).to(device=param.device, dtype=param.dtype)  # [1,3,H,W]

            # ====== Calculate effective prefill length and trim max_new_tokens (before generate)======
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
                    input_ids = input_ids[:, -keep:]  # Trim from the right to keep the end
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
                # Make it conservative again: ensure position limit - existing token - at least 1 token
                safe_max_new = min(safe_max_new, max(0, int(max_pos or 4096) - input_ids.size(1) - 1))
            else:
                safe_max_new = args.max_new_tokens

            # ====== Actual generation (no image_sizes passed; decode only new tokens)======
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

            # Decode only the newly generated tokens, avoid decoding the prompt
            gen_only = output_ids[:, input_len:]
            outputs = tokenizer.batch_decode(gen_only, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0].strip()

            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({"question_id": idx,
                                    "round_id": round_idx,
                                    "prompt": cur_prompt,
                                    "text": outputs,
                                    "options": options,
                                    "option_char": cur_option_char,
                                    "answer_id": ans_id,
                                    "model_id": model_name,
                                    "metadata": {}}) + "\n")
            ans_file.flush()

            # Rotate options
            options = options[1:] + options[:1]
            cur_option_char = cur_option_char[1:] + cur_option_char[:1]
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
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--all-rounds", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--max-new-tokens", type=int, default=128)

    args = parser.parse_args()

    eval_model(args)
