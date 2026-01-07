#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="$PWD:$PYTHONPATH"
echo "PYTHONPATH=$PYTHONPATH"

SPLIT="mmbench_dev_20230712"

python -m run.eval.mmbench.model_vqa_mmbench \
    --model-path ./output/llava_fgvpo/llava_fgvpo_merged \
    --question-file /path/to/your/project/evaluation/mmbench/${SPLIT}.tsv \
    --answers-file ./output/eval/mmbench/answers/${SPLIT}/llava_fgvpo_merged.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python run/eval/mmbench/convert_mmbench_for_submission.py \
    --annotation-file /path/to/your/project/evaluation/mmbench/${SPLIT}.tsv \
    --result-dir ./output/eval/mmbench/answers/${SPLIT} \
    --upload-dir ./output/eval/mmbench/answers_upload/${SPLIT} \
    --experiment llava_fgvpo_merged

python run/eval/mmbench/score_mmbench_dev.py \
    --tsv /path/to/your/project/evaluation/mmbench/${SPLIT}.tsv \
    --pred ./output/eval/mmbench/answers/${SPLIT}/llava_fgvpo_merged.jsonl
