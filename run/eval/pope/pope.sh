#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python -m run.eval.pope.pope_merged_all.model_vqa_loader \
    --model-path ./output/llava_fgvpo/llava_fgvpo_merged \
    --question-file /path/to/your/project/evaluation/pope/llava_pope_test.jsonl \
    --image-folder /path/to/your/project/dataset/MSCOCO/val2014 \
    --answers-file ./output/eval/pope/llava_fgvpo_merged.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_pope.py \
    --annotation-dir /path/to/your/project/evaluation/pope/coco \
    --question-file /path/to/your/project/evaluation/pope/llava_pope_test.jsonl \
    --result-file ./output/eval/pope/llava_fgvpo_merged.jsonl 


