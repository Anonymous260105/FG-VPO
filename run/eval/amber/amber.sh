#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PWD:$PYTHONPATH"
echo "PYTHONPATH=$PYTHONPATH"

MODEL_BASE=./output/llava_fgvpo/llava_fgvpo_merged
MODEL_SUFFIX=llava_fgvpo_merged

IMAGE_DIR_AMBER=/path/to/your/project/evaluation/AMBER/image
QUESTION_FILE=/path/to/your/project/evaluation/AMBER/query/query_generative.json

OUTPUT_DIR=./output/eval/amber/amber_AMBER_llava_fgvpo_merged

python -m run.eval.amber.amber_generate \
    --model-path ${MODEL_BASE} \
    --temperature 0.0 \
    --answers-file ${OUTPUT_DIR}/answer/answer-file-${MODEL_SUFFIX}.jsonl \
    --image-file "${IMAGE_DIR_AMBER}" \
    --question-file "${QUESTION_FILE}" \
    --image_aspect_ratio pad \
    --test-prompt ''

python -m run.eval.amber.amber_eval \
    --inference_data ${OUTPUT_DIR}/answer/answer-file-${MODEL_SUFFIX}.jsonl \
    --evaluation_type g