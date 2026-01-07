import sys
import os
import json
import shutil
import argparse
import torch

# 添加 llava 模块路径（保留你的做法）
sys.path.insert(0, '/home/lijingran/scheme/OPA-DPO')

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from peft import PeftModel

def _exists(p): 
    return p is not None and os.path.exists(p)

def _first_exist(dirpath, names):
    for n in names:
        p = os.path.join(dirpath, n)
        if os.path.exists(p):
            return p
    return None

def _copy_if_exists(src_dir, dst_dir, names):
    os.makedirs(dst_dir, exist_ok=True)
    copied = []
    for n in names:
        p = os.path.join(src_dir, n)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(dst_dir, n))
            copied.append(n)
    return copied

def _load_json(p):
    with open(p, "r") as f:
        return json.load(f)

def _merge_llava_config(base_cfg, lora_cfg):
    """把 LoRA 里和多模态/位置编码相关的字段覆盖到 base 的 config 上。"""
    keys = [
        "mm_projector_type", "mm_hidden_size", "mm_vision_tower",
        "mm_use_im_start_end", "image_token_index",
        "mm_patch_merge_type", "vision_resolution", "vision_tower_cfg",
        "rope_scaling"  # 如果训练时用了 rope 扩展（非常关键）
    ]
    out = dict(base_cfg)
    for k in keys:
        if k in lora_cfg:
            out[k] = lora_cfg[k]
    # 容错：若训练时用了 IM_START/END，但 lora cfg 里没写，显式置 True
    if "mm_use_im_start_end" not in out:
        out["mm_use_im_start_end"] = True
    return out

def merge_lora(args):
    print("=" * 50)
    print("LLaVA-LoRA Model Merger")
    print("=" * 50)
    print(f"Using CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'all available')}")

    # 1) 路径检查
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Base model path not found: {args.model_path}")
    if not os.path.exists(args.lora_model_path):
        raise FileNotFoundError(f"LoRA model path not found: {args.lora_model_path}")
    os.makedirs(args.save_model_path, exist_ok=True)

    # 2) 识别 LoRA 必备文件（bin 或 safetensors 二选一）
    adapter_model_path = _first_exist(args.lora_model_path, ["adapter_model.safetensors", "adapter_model.bin"])
    adapter_cfg_path   = os.path.join(args.lora_model_path, "adapter_config.json")
    if not _exists(adapter_model_path):
        raise FileNotFoundError(
            f"LoRA weights not found under {args.lora_model_path}. "
            f"Expect one of: adapter_model.safetensors / adapter_model.bin"
        )
    if not _exists(adapter_cfg_path):
        raise FileNotFoundError(f"adapter_config.json not found: {adapter_cfg_path}")
    print(f"✓ Found LoRA files: {os.path.basename(adapter_model_path)}, adapter_config.json")

    # 3) 加载底座（llava 的包装器会把视觉塔/投影等一起构建）
    print("\n1. Loading base LLaVA model...")
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path,
        args.model_base,
        model_name
    )

    # 可选：把模型放到 GPU 并转半精（很多 LLaVA 权重默认就是 fp16，转一下更稳）
    if torch.cuda.is_available():
        model = model.to("cuda")
    try:
        model = model.half()
    except Exception:
        pass
    
    print("✓ Base model loaded")

    # 4) 载入 LoRA 并 merge
    print("\n2. Loading LoRA adapter & merging...")
    model = PeftModel.from_pretrained(model, args.lora_model_path)
    # 关键：合并并卸载 LoRA 适配器
    model = model.merge_and_unload()
    print("✓ LoRA merged into base")

    # 合并完成后，强制半精 + 推理模式
    try:
        model.to(dtype=torch.float16)
    except Exception:
        pass
    model.eval()
    torch.set_grad_enabled(False)

    # 明确告诉配置要用快速注意力 & 半精
    # 优先 flash-attn2，其次 sdpa（按你环境支持）
    setattr(model.config, "use_cache", True)
    if hasattr(model.config, "attn_implementation"):
        model.config.attn_implementation = "flash_attention_2"  # 或 "sdpa"
    else:
        # 新版 transformers 只认 attn_implementation；旧版可能是别名/flag
        try:
            model.config.update({"attn_implementation": "flash_attention_2"})
        except Exception:
            pass

    # 标注 dtype，避免下次 from_pretrained 回到 fp32
    try:
        model.config.torch_dtype = "float16"
    except Exception:
        pass


    # 5) 保存合并后的模型与分词器
    print(f"\n3. Saving merged model to {args.save_model_path} ...")
    model.save_pretrained(args.save_model_path, safe_serialization=True)
    tokenizer.save_pretrained(args.save_model_path)
    if image_processor is not None:
        try:
            image_processor.save_pretrained(args.save_model_path)
            print("✓ Image processor saved")
        except Exception as e:
            print(f"⚠ Could not save image processor: {e}")

    # 6) 多模态必需文件从 LoRA 目录拷贝到融合目录
    #    常见：projector、preprocessor_config、special_tokens_map
    copied = _copy_if_exists(
        args.lora_model_path, args.save_model_path,
        [
            "mm_projector.bin",
            "mm_projector.safetensors",
            "preprocessor_config.json",
            "special_tokens_map.json",
            "tokenizer_config.json"  # 有些 LoRA 里也会带这个
        ]
    )
    if copied:
        print("✓ Copied MM files from LoRA:", ", ".join(copied))
    else:
        print("⚠ No extra MM files found to copy (this may be fine if base already contains them)")

    # 7) 合并 config.json：以 LoRA 的多模态字段覆盖 base
    base_cfg_path = os.path.join(args.model_path, "config.json")
    lora_cfg_path = os.path.join(args.lora_model_path, "config.json")
    out_cfg_path  = os.path.join(args.save_model_path, "config.json")

    # 此时 out_cfg 已由 model.save_pretrained() 写好，里面保留了 torch_dtype/attn_implementation 等关键信息
    saved_cfg = _load_json(out_cfg_path)

    if _exists(lora_cfg_path):
        lora_cfg = _load_json(lora_cfg_path)
        # 只更新多模态字段，避免覆盖 dtype/注意力实现等加速选项
        for k in [
            "mm_projector_type","mm_hidden_size","mm_vision_tower",
            "mm_use_im_start_end","image_token_index",
            "mm_patch_merge_type","vision_resolution","vision_tower_cfg",
            "rope_scaling"
        ]:
            if k in lora_cfg:
                saved_cfg[k] = lora_cfg[k]

        with open(out_cfg_path, "w") as f:
            json.dump(saved_cfg, f, indent=2, ensure_ascii=False)
        print("✓ Updated MM fields while preserving dtype/attention settings")
    else:
        print("⚠ LoRA config.json not found; kept saved config.json as-is")


    print("\n" + "=" * 50)
    print("Model merging completed successfully!")
    print(f"Merged model saved to: {args.save_model_path}")
    print("=" * 50)

    print("[Post-Merge Sanity]")
    print("  - dtype:", next(model.parameters()).dtype)
    print("  - attn_impl:", getattr(model.config, "attn_implementation", None))
    print("  - use_cache:", getattr(model.config, "use_cache", None))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LLaVA base model with LoRA weights")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the base LLaVA model (e.g., llava_7b)")
    parser.add_argument("--model-base", type=str, default=None,
                        help="Base model name (optional)")
    parser.add_argument("--lora-model-path", type=str, required=True,
                        help="Path to LoRA checkpoint directory")
    parser.add_argument("--save-model-path", type=str, required=True,
                        help="Path to save the merged model")
    args = parser.parse_args()
    merge_lora(args)
