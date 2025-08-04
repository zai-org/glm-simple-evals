# GLM-SIMPLE-EVALS

[中文版](./README_zh.md)

GLM-SIMPLE-EVALS is an internal evaluation toolset for large language models developed by Zhipu AI, based on OpenAI's [simple-evals](https://github.com/openai/simple-evals) project. We have open-sourced it to allow the community to reproduce the performance of Zhipu AI's officially released GLM-4.5 model on various evaluation metrics.

## Supported Evaluation Tasks

Currently, this repository supports the following evaluation tasks, covering multiple domains such as reasoning, coding, and mathematics:

- AIME24
- GPQA
- HLE
- LiveCodeBench
- MATH 500
- SciCode
- MMLU Pro

## Supported Model Calling Methods

This repository supports two model calling methods:

- Zhipu's official `zai-sdk`
- Model calls compatible with OpenAI's interface

## Quick Start

We provide an example script `eval_example.sh`. You only need to configure the `api_key` and other necessary parameters (such as model address, verification model address, etc.) to start the evaluation.

### Task-specific Usage Guides

#### 1. HLE

In the HLE evaluation task, you need to use `gpt-4o` to verify the results. Execute the following command to perform the evaluation:

```bash
python3 evaluate.py \
    --model_name "glm-4.5" \
    --backbone "zai" \
    --zai_api_key "xxxxxx" \
    --save_dir "/temp/eval_results" \
    --tasks hle \
    --proc_num 60 \
    --auto_extract_answer \
    --max_new_tokens 65536 \
    --checker_model_name "gpt-4o" \
    --checker_url "xxxx" \ # If empty, points to OpenAI's official interface
    --checker_api_key "xxxx" \
    --stream \
```

#### 2. LiveCodeBench

In the LiveCodeBench evaluation task, you need to specify the test date as `2407_2501`. Execute the following command to perform the evaluation:

```bash
python3 evaluate.py \
    --model_name "glm-4.5" \
    --backbone "zai" \
    --zai_api_key "xxxxxx" \
    --save_dir "/temp/eval_results" \
    --tasks lcb \
    --lcb_date "2407_2501" \
    --proc_num 60 \
    --auto_extract_answer \
    --max_new_tokens 65536 \
    --stream \
```

#### 3. Other Evaluation Tasks
For other evaluation tasks (AIME24, GPQA, MATH 500, SciCode, MMLU Pro), the verification model uses `Meta-Llama-3.1-70B-Instruct`. Execute the following command to perform the evaluation:
```bash
python3 evaluate.py \
    --model_name "glm-4.5" \
    --backbone "zai" \
    --zai_api_key "xxxxxx" \
    --save_dir "/temp/eval_results" \
    --tasks aime2024 \  # gpqa math500 mmlu_pro scicode
    --proc_num 60 \
    --checker_model_name "Meta-Llama-3.1-70B-Instruct" \
    --checker_url "xxxxx" \
    --auto_extract_answer \
    --max_new_tokens 65536 \
    --stream \
```

## Parameter Description
The following are descriptions of commonly used parameters in the evaluation script:
- `--model_name`: Name of the model to be evaluated
- `--backbone`: Model calling method, supporting "zai" (Zhipu's official SDK) or other methods compatible with OpenAI's interface
- `--zai_api_key`: API Key for Zhipu BigModel platform
- `--save_dir`: Directory to save evaluation results
- `--tasks`: Evaluation tasks
- `--proc_num`: Number of concurrent processes
- `--auto_extract_answer`: Automatically extract answers
- `--max_new_tokens`: Maximum number of tokens for generated text
- `--checker_model_name`: Name of the verification model
- `--checker_url`: API address of the verification model
- `--checker_api_key`: API key of the verification model
- `--stream`: Whether to use streaming output
- `--lcb_date`: Test date range for LiveCodeBench evaluation

## Important Notes
1. Please ensure you have obtained the necessary API keys and have correctly configured them in the evaluation script.
2. Depending on different evaluation tasks, specific verification models may need to be configured.
3. Evaluation results will be saved in the specified `--save_dir` directory.
4. Please adjust the `--proc_num` parameter appropriately according to your hardware resources to achieve the best evaluation efficiency.