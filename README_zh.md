# GLM-SIMPLE-EVALS

GLM-SIMPLE-EVALS 是智谱AI内部使用的大模型评测工具集，基于 OpenAI 的 [simple-evals](https://github.com/openai/simple-evals) 项目开发。我们将其开源，以便社区能够复现智谱AI官方发布的 GLM-4.5 模型在各项评测指标上的表现。

## 支持的评测任务

目前，本仓库支持以下评测任务，涵盖推理、代码、数学等多个领域：

- AIME
- GPQA
- HLE
- LiveCodeBench
- MATH 500
- SciCode
- MMLU Pro

## 支持的模型调用方式

本仓库支持两种模型调用方式：
- 智谱官方 `zai-sdk`
- 兼容 OpenAI 接口的模型调用

## 快速开始

我们提供了 `eval_example.sh` 示例脚本，您只需配置其中的 `api_key` 和其他必要参数（如模型地址、校验模型地址等），并下载完所需数据即可开始评测。

### 环境配置

我们建议使用`python==3.10`环境。并运行如下命令配置该仓库所必需的环境依赖。

```bash
pip install -r requirements.txt
```

### 评测数据下载指南

1. 下载 [glm-simple-evals-dataset](https://huggingface.co/datasets/zai-org/glm-simple-evals-dataset)，并将其放置在 `./data` 目录下。

2. 下载 SciCode 所需测试用例，该数据集来源于 [SciCode 官方仓库](https://github.com/scicode-bench/SciCode/tree/main)。请从 [Google Drive 链接](https://drive.google.com/drive/folders/1W5GZW6_bdiDAiipuFMqdUhvUaHIj6-pR?usp=drive_link) 下载测试数据，并将其放置在 `./data/scicode/test_data.h5`。**注意**：在使用该数据集时，请遵守其原始仓库中规定的许可证条款和使用条件。

### 评测任务使用指南

#### 1. HLE

在 HLE 评测任务中，需要使用 `gpt-4o` 对结果进行校验。执行以下命令进行评测：

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
    --checker_url "xxxx" \ # 如果为空，则指向OpenAI官方接口
    --checker_api_key "xxxx" \
    --stream \
```

#### 2. LiveCodeBench

在 LiveCodeBench 评测任务中，需要指定测试日期为 `2407_2501`。执行以下命令进行评测：

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

#### 3. 其他评测任务

对于其他评测任务（AIME、GPQA、MATH 500、SciCode、MMLU Pro），校验模型采用 `Meta-Llama-3.1-70B-Instruct`。执行以下命令进行评测：

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

## 参数说明

以下是评测脚本中常用参数的说明：

- `--model_name`: 待评测的模型名称
- `--backbone`: 模型调用方式，支持 "zai"（智谱官方SDK）或其他兼容OpenAI接口的方式
- `--zai_api_key`: 智谱BigModel平台的API Key
- `--save_dir`: 评测结果保存目录
- `--tasks`: 评测任务
- `--proc_num`: 并发进程数
- `--auto_extract_answer`: 自动提取答案
- `--max_new_tokens`: 生成文本的最大token数
- `--checker_model_name`: 校验模型名称
- `--checker_url`: 校验模型API地址
- `--checker_api_key`: 校验模型API密钥
- `--stream`: 是否使用流式输出
- `--lcb_date`: LiveCodeBench评测的测试日期范围

## 注意事项

1. 请确保您已获取必要的API密钥，并已正确配置在评测脚本中。
2. 根据不同的评测任务，可能需要配置特定的校验模型。
3. 评测结果将保存在指定的 `--save_dir` 目录中。
4. 请根据您的硬件资源适当调整 `--proc_num` 参数，以获得最佳的评测效率。