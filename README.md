# GLM-SIMPLE-EVALS

[中文版](./README_zh.md)

GLM-SIMPLE-EVALS is an internal evaluation toolset for large language models developed by Z.ai, based on OpenAI's [simple-evals](https://github.com/openai/simple-evals) project. We have open-sourced it to allow the community to reproduce the performance of Z.ai's officially released GLM-4.5 model on various evaluation metrics.

## Supported Evaluation Tasks

Currently, this repository supports the following evaluation tasks, covering multiple domains such as reasoning, coding, and mathematics:

- AIME
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

### Environment Setup

We recommend using a `python==3.10` environment. Run the following command to configure the necessary dependencies for this repository.

```bash
pip install -r requirements.txt
```

### Download the Evaluation Data 

1. Download the [glm-simple-evals-dataset](https://huggingface.co/datasets/zai-org/glm-simple-evals-dataset) and place it in the `./data` directory.

2. Download the test cases required for SciCode, which originates from the [SciCode official repository](https://github.com/scicode-bench/SciCode/tree/main). Please download the test data from the [Google Drive link](https://drive.google.com/drive/folders/1W5GZW6_bdiDAiipuFMqdUhvUaHIj6-pR?usp=drive_link) and place it at `./data/scicode/test_data.h5`. **Note**: When using this dataset, please comply with the license terms and usage conditions specified in its original repository.

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
    --max_new_tokens 81920 \
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
    --max_new_tokens 81920 \
    --stream \
```

#### 3. Other Evaluation Tasks
For other evaluation tasks (AIME, GPQA, MATH 500, SciCode, MMLU Pro), the verification model uses `Meta-Llama-3.1-70B-Instruct`. Execute the following command to perform the evaluation:
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
    --max_new_tokens 81920 \
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