#!/bin/bash

export CUDA_LAUNCH_BLOCKING=0

python -m run.eval.textvqa.model_vqa_loader \
    --model-path ./output/llava_fgvpo/llava_fgvpo_merged \
    --question-file /path/to/your/project/evaluation/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder /path/to/your/project/evaluation/textvqa/train_images \
    --answers-file ./output/eval/textvqa/answers/llava_fgvpo_merged.jsonl\
    --temperature 0 \
    --conv-mode vicuna_v1

python -m run.eval.textvqa.eval_textvqa \
    --annotation-file /path/to/your/project/evaluation/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./output/eval/textvqa/answers/llava_fgvpo_merged.jsonl
