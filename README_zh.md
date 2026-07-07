# GLM-Simple-Evals

GLM-Simple-Evals 是智谱 AI 内部使用的大语言模型评测工具集，基于 OpenAI 的 [simple-evals](https://github.com/openai/simple-evals) 项目开发。我们将此工具开源，以帮助开发者社区复现智谱 AI 发布的 GLM-4.5 和 GLM-4.6 模型在各评测任务中的表现。

## 评测任务支持

当前支持以下评测任务，覆盖推理、代码、数学等多个领域：

- AIME
- GPQA
- HLE
- LiveCodeBench
- MATH 500
- SciCode
- MMLU Pro
- PerspectiveGap（`role_assignment`、`prompt_writing`）

## 模型接入方式

本项目支持以下两种模型调用方式：
- 智谱官方 `zai-sdk`
- 兼容 OpenAI 接口的模型调用

## 快速开始

我们提供了 `eval_example.sh` 示例脚本，您只需配置其中的 API 密钥和其他必要参数（如模型地址、校验模型地址等），并下载所需数据即可开始评测。

具体评测指南请参考以下文档：

- **[GLM-4.6](./news/glm-4.6/GLM4.6_zh.md)**
- **[GLM-4.5 & GLM-4.5-Air](./news/glm-4.5/GLM4.5_zh.md)**

### 环境要求

推荐使用 Python 3.10 环境。运行以下命令安装所需依赖：

```bash
pip install -r requirements.txt
```

### 数据准备

1. 下载 [glm-simple-evals-dataset](https://huggingface.co/datasets/zai-org/glm-simple-evals-dataset)，并将其放置在 `./data` 目录下。

2. 下载 SciCode 所需测试用例，该数据集来源于 [SciCode 官方仓库](https://github.com/scicode-bench/SciCode/tree/main)。请从 [Google Drive 链接](https://drive.google.com/drive/folders/1W5GZW6_bdiDAiipuFMqdUhvUaHIj6-pR?usp=drive_link) 下载测试数据，并将其放置在 `./data/scicode/test_data.h5`。**注意**：在使用该数据集时，请遵守其原始仓库中规定的许可证条款和使用条件。

3. PerspectiveGap：`requirements.txt` 已包含官方评分器，并固定到 commit `9c6921b3337ff3e6a6a453f68d117a8c1663135e`。如需单独安装，可运行 `pip install 'perspective-gap @ git+https://github.com/WhymustIhaveaname/PerspectiveGap.git@9c6921b3337ff3e6a6a453f68d117a8c1663135e'`。如果没有本地文件，评测器会通过 `datasets` 库加载 [`sun1245/PerspectiveGap`](https://huggingface.co/datasets/sun1245/PerspectiveGap)；离线运行时，可将官方 JSONL 放到 `./data/perspective_gap/evaluations.jsonl`。

### 使用指南

#### 1. HLE

在 HLE 评测任务中，需要使用 `gpt-4o` 对结果进行校验。如不设置 `--checker_url`，默认使用 OpenAI 官方接口。执行以下命令进行评测：

```bash
python3 evaluate.py \
    --model_name "glm-4.6" \
    --backbone "zai" \
    --zai_api_key "xxxxxx" \
    --save_dir "./eval_results" \
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

在 LiveCodeBench 评测任务中，需要指定测试日期为 `2407_2501`。执行以下命令进行评测：

```bash
python3 evaluate.py \
    --model_name "glm-4.6" \
    --backbone "zai" \
    --zai_api_key "xxxxxx" \
    --save_dir "./eval_results" \
    --tasks lcb \
    --lcb_date "2407_2501" \
    --proc_num 60 \
    --auto_extract_answer \
    --max_new_tokens 128000 \
    --stream \
    --top_p 0.95 \
```

#### 3. 其他评测任务

对于其他评测任务（AIME、GPQA、MATH 500、SciCode、MMLU Pro），校验模型采用 `Meta-Llama-3.1-70B-Instruct`。执行以下命令进行评测：

```bash
python3 evaluate.py \
    --model_name "glm-4.6" \
    --backbone "zai" \
    --zai_api_key "xxxxxx" \
    --save_dir "./eval_results" \
    --tasks aime2024 \  # gpqa math500 mmlu_pro scicode
    --proc_num 60 \
    --checker_model_name "Meta-Llama-3.1-70B-Instruct" \
    --checker_url "xxxxx" \
    --auto_extract_answer \
    --max_new_tokens 128000 \
    --stream \
```

#### 4. PerspectiveGap

这里以社区参考任务的形式接入 [PerspectiveGap](https://arxiv.org/abs/2606.08878) 的两个任务，用于评测多智能体编排提示和角色级上下文路由：`role_assignment` 和 `prompt_writing`。这两个任务使用官方 `perspective_gap.scoring` 确定性评分器，不需要校验模型。

```bash
python3 evaluate.py \
    --model_name "glm-4.6" \
    --backbone "zai" \
    --zai_api_key "xxxxxx" \
    --save_dir "./eval_results" \
    --tasks role_assignment prompt_writing \
    --proc_num 20 \
    --max_new_tokens 8192 \
    --stream
```

若使用 OpenAI 兼容接口，请照常设置 `--backbone openai`、`--openai_api_key` 和 `--openai_base_url`。

## 参数说明

以下是评测脚本中常用参数的说明：

- `--model_name`: 待评测的模型名称
- `--backbone`: 模型调用方式，`zai`（智谱官方 SDK）或 `openai`（OpenAI 兼容接口）
- `--zai_api_key`: 智谱 BigModel 平台 API Key
- `--openai_api_key`: OpenAI 兼容服务的 API Key
- `--openai_base_url`: OpenAI 兼容服务的 API 地址
- `--save_dir`: 评测结果保存目录
- `--tasks`: 评测任务
- `--proc_num`: 并发进程数
- `--auto_extract_answer`: 自动提取答案
- `--max_new_tokens`: 生成文本的最大 Token 数
- `--checker_model_name`: 校验模型名称
- `--checker_url`: 校验模型 API 地址
- `--checker_api_key`: 校验模型 API Key
- `--stream`: 是否使用流式输出
- `--lcb_date`: LiveCodeBench 评测的测试日期范围，可选 `2407_2501` 或 `v6`

## 注意事项

1. 请确保已获取必要的 API Key，并在评测脚本中正确配置。
2. 根据不同的评测任务，可能需要配置特定的校验模型。
3. 评测结果将保存在指定的 `--save_dir` 目录中。
4. 请根据硬件资源和接口限流适当调整 `--proc_num` 参数。
