# GLM-Simple-Evals

[中文版](./README_zh.md)

GLM-Simple-Evals is an internal evaluation toolkit for large language models developed by Z.AI, based on OpenAI's [simple-evals](https://github.com/openai/simple-evals) project. We have open-sourced this framework to help the developer community reproduce the performance of Z.AI's released GLM-4.5 and GLM-4.6 models across various evaluation tasks.

## Supported Evaluation Tasks

Currently supports the following evaluation benchmarks, covering reasoning, coding, mathematics, and other domains:

- AIME
- GPQA
- HLE
- LiveCodeBench
- MATH 500
- SciCode
- MMLU Pro
- PerspectiveGap (`role_assignment`, `prompt_writing`)

## Model Integration

This project supports the following model inference methods:

- Z.AI official `zai-sdk`
- OpenAI-compatible API interface

## Quick Start

We provide an `eval_example.sh` example script. You only need to configure the API key and other necessary parameters (such as model endpoints, verification model endpoints, etc.) and download the required data to begin evaluation.

For detailed evaluation guidelines, please refer to the following documentation:

- **[GLM-4.6](./news/glm-4.6/GLM4.6.md)**
- **[GLM-4.5 & GLM-4.5-Air](./news/glm-4.5/GLM4.5.md)**

### Requirements

We recommend using Python 3.10. Install the required dependencies with:

```bash
pip install -r requirements.txt
```

### Data Preparation

1. Download the [glm-simple-evals-dataset](https://huggingface.co/datasets/zai-org/glm-simple-evals-dataset) and place it in the `./data` directory.

2. Download SciCode test cases from the [SciCode official repository](https://github.com/scicode-bench/SciCode/tree/main). Download the test data from the [Google Drive link](https://drive.google.com/drive/folders/1W5GZW6_bdiDAiipuFMqdUhvUaHIj6-pR?usp=drive_link) and place it at `./data/scicode/test_data.h5`. **Note**: Please comply with the license terms and usage conditions specified in the original repository when using this dataset.

3. For PerspectiveGap, either allow the evaluator to load [`sun1245/PerspectiveGap`](https://huggingface.co/datasets/sun1245/PerspectiveGap) through `datasets`, or place the official JSONL file at `./data/perspective_gap/evaluations.jsonl`. Install the official scorer with `pip install 'perspective-gap @ git+https://github.com/WhymustIhaveaname/PerspectiveGap.git'`.

### Usage Guide

#### 1. HLE

In the HLE evaluation task, `gpt-4o` is required for result verification. Execute the following command to run the evaluation:

```bash
python3 evaluate.py \
    --model_name "glm-4.6" \
    --backbone "zai" \
    --zai_api_key "xxxxxx" \
    --save_dir "/temp/eval_results" \
    --tasks hle \
    --proc_num 60 \
    --auto_extract_answer \
    --max_new_tokens 128000 \
    --checker_model_name "gpt-4o" \
    --checker_url "xxxx" \
    --checker_api_key "xxxx" \
    --stream \
```

#### 2. LiveCodeBench

In the LiveCodeBench evaluation task, you need to specify the test date range as `2407_2501`. Execute the following command to run the evaluation:

```bash
python3 evaluate.py \
    --model_name "glm-4.6" \
    --backbone "zai" \
    --zai_api_key "xxxxxx" \
    --save_dir "/temp/eval_results" \
    --tasks lcb \
    --lcb_date "2407_2501" \
    --proc_num 60 \
    --auto_extract_answer \
    --max_new_tokens 128000 \
    --stream \
    --top_p 0.95 \
```

#### 3. Other Benchmarks

For other evaluation tasks (AIME, GPQA, MATH 500, SciCode, MMLU Pro), the verification model uses `Meta-Llama-3.1-70B-Instruct`. Execute the following command to run the evaluation:

```bash
python3 evaluate.py \
    --model_name "glm-4.6" \
    --backbone "zai" \
    --zai_api_key "xxxxxx" \
    --save_dir "/temp/eval_results" \
    --tasks aime2024 \  # gpqa math500 mmlu_pro scicode
    --proc_num 60 \
    --checker_model_name "Meta-Llama-3.1-70B-Instruct" \
    --checker_url "xxxxx" \
    --auto_extract_answer \
    --max_new_tokens 128000 \
    --stream \
```

#### 4. PerspectiveGap

PerspectiveGap provides two community reference tasks for multi-agent orchestration prompting: `role_assignment` and `prompt_writing`. These tasks use the deterministic official `perspective_gap.scoring` metrics and do not require a checker model.

```bash
python3 evaluate.py \
    --model_name "glm-4.6" \
    --backbone "zai" \
    --zai_api_key "xxxxxx" \
    --save_dir "/temp/eval_results" \
    --tasks role_assignment prompt_writing \
    --proc_num 20 \
    --max_new_tokens 8192 \
    --stream
```

For OpenAI-compatible endpoints, use `--backbone openai`, `--openai_api_key`, and `--openai_base_url` as usual.

## Parameter Reference

The following are descriptions of commonly used parameters in the evaluation script:

- `--model_name`: Target model name for evaluation
- `--backbone`: Model inference method, "zai" (Z.AI official SDK) or "openai" (OpenAI-compatible APIs)
- `--zai_api_key`: API key for Z.AI API platform
- `--openai_api_key`: API key for other OpenAI-compatible providers
- `--openai_base_url`: API URL for other OpenAI-compatible providers
- `--save_dir`: Output directory for evaluation results
- `--tasks`: Evaluation benchmarks to run
- `--proc_num`: Number of concurrent processes
- `--auto_extract_answer`: Enable automatic answer extraction
- `--max_new_tokens`: Maximum number of generation tokens
- `--checker_model_name`: Verification model name
- `--checker_url`: API endpoint for verification model
- `--checker_api_key`: API key for verification model
- `--stream`: Enable streaming output
- `--lcb_date`: Test date range for LiveCodeBench evaluation, supports '2407_2501' or 'v6'

## Parameter Reference

1. Ensure you have obtained the necessary API keys and have properly configured them in the evaluation script.
2. Different evaluation tasks may require specific verification models.
3. Evaluation results will be saved in the specified `--save_dir` directory.
4. Please adjust the `--proc_num` parameter according to your hardware resources to achieve optimal evaluation efficiency.