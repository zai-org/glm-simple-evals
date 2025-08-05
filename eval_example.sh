#!/bin/bash


MODEL_NAME="glm-4.5"  # or "glm-4.5-air"

#For zai-sdk
BACKBONE="zai"
ZAI_API_KEY="Your API key from https://bigmodel.cn/"
OPENAI_BASE_URL="" # The base url of the openai api need to be set when using openai-sdk, zai-sdk is not needed
OPENAI_API_KEY="" # The api key of the openai api need to be set when using openai-sdk, zai-sdk is not needed

# For openai-sdk
# BACKBONE="openai"
# OPENAI_BASE_URL="https://open.bigmodel.cn/api/paas/v4/" # The base url of the openai api
# OPENAI_API_KEY="Your API key from https://bigmodel.cn/" # The api key of the openai api

CHECKER_URL="The checker model's url"  # Like http://0.0.0.0:8000/v1
CHECEKR_MODEL_NAME="The checker model's name"  # "Meta-Llama-3.1-70B-Instruct" or "gpt-4o"

SAVE_DIR="The save path of the evaluation results"
PROC_NUM=60 # The number of processes to run the evaluation

MAX_NEW_TOKENS=81920 # The max new tokens of the evaluation

python3 evaluate.py \
--model_name $MODEL_NAME \
--backbone $BACKBONE \
--zai_api_key $ZAI_API_KEY \
--openai_api_key $OPENAI_API_KEY \
--openai_base_url $OPENAI_BASE_URL \
--save_dir $SAVE_DIR \
--tasks aime2024 \
--proc_num $PROC_NUM \
--checker_model_name $CHECEKR_MODEL_NAME \
--checker_url $CHECKER_URL \
--auto_extract_answer \
--max_new_tokens $MAX_NEW_TOKENS \
--stream \