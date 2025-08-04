import re
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue
from typing import Any
from tqdm import tqdm

from utils.types import EvalResult, SingleEvalResult, SamplerBase

MATH_QUERY_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form ANSWER: $ANSWER (without quotes) where $ANSWER is the answer to the problem.

{Question}

Remember to put your answer on its own line after "ANSWER:", and you do not need to use a \\boxed command.
""".strip()


ANSWER_PATTERN = r"(?i)ANSWER\s*:\s*([^\n]+)"

EQUALITY_TEMPLATE = r"""
Look at the following two expressions (answers to a math problem) and judge whether they are equivalent. Only perform trivial simplifications
and considering the give prolem to solve

Problem: {question}

Examples:

    Expression 1: $2x+3$
    Expression 2: $3+2x$

Yes

    Expression 1: 3/2
    Expression 2: 1.5

Yes

    Expression 1: $x^2+2x+1$
    Expression 2: $y^2+2y+1$

No

    Expression 1: $x^2+2x+1$
    Expression 2: $(x+1)^2$

Yes

    Expression 1: 3245/5
    Expression 2: 649

No
(these are actually equal, don't mark them equivalent if you need to do nontrivial simplifications)

    Expression 1: 2/(-3)
    Expression 2: -2/3

Yes
(trivial simplifications are allowed)

    Expression 1: 72 degrees
    Expression 2: 72

Yes
(give benefit of the doubt to units)

    Expression 1: 64
    Expression 2: 64 square feet

Yes
(give benefit of the doubt to units)

    Expression 1: \\text{[number]}
    Expression 2: 64

No

    Expression 1: 2
    Expression 2: [number]

No

    Expression 1: b = 2
    Expression 2: 2

Yes


---

YOUR TASK


Respond with only "Yes" or "No" (without quotes). Do not include a rationale.

    Expression 1: %(expression1)s
    Expression 2: %(expression2)s
""".strip()

def check_equality(sampler: SamplerBase, expr1: str, expr2: str, question: str, qwq_check: bool):
    prompt = EQUALITY_TEMPLATE % {"expression1": expr1, "expression2": expr2, "question": question}

    for _ in range(3):
        response = sampler([dict(content=prompt, role="user")])
        if len(response) == 0:
            continue
        else:
            break
    if len(response) == 0:
        return None
    if '[3, +\\infty)' in expr1 or '[3, +\\infty)' in expr2:
        print(response)
    # print(response)
    if "<answer>" in response and "</answer>" in response:
        response = re.findall(r"<answer>([\s\S]+?)</answer>", response)[-1].strip()
    elif qwq_check:
        pattern = r'</think>(.*)'  # qwq 答案截断
        match = re.search(pattern, response, re.DOTALL)
        if match:
            response = match.group(1)  # 返回匹配到的'</think>'之后的内容
        else:
            response = ""  # 如果没有找到'</think>'，返回空字符串
    return response.lower().strip() == "yes"

def get_final_answer(sampler, equality_checker, question, auto_extract_answer=False):
    if auto_extract_answer:
        prompt_messages = [dict(content=question, role="user")]
        response = sampler(prompt_messages)
        if '<think>' in response:
            if '</think>' in response:
                response = response.split('</think>')[1]
            else:
                return  None, response
        extracted_answer = extract_answer(equality_checker, question, response)
    else:
        prompt_messages = [dict(content=MATH_QUERY_TEMPLATE.format(Question=question), role="user")]
        response = sampler(prompt_messages)
        match = re.search(ANSWER_PATTERN, response)
        extracted_answer = match.group(1) if match else None
    return extracted_answer, response


def _compute_stat(values: list, stat: str):
    if stat == "mean":
        return np.mean(values)
    elif stat == "std":
        return np.std(values)
    elif stat == "min":
        return np.min(values)
    elif stat == "max":
        return np.max(values)
    else:
        raise ValueError(f"Unknown {stat =}")


def aggregate_results(
    single_eval_results: list[SingleEvalResult],
    default_stats: tuple[str] = ("mean", "std"),
    name2stats: dict[str, tuple[str]] = None
) -> EvalResult:
    """
    Aggregate results from multiple evaluations into a single EvalResult.
    """
    name2stats = name2stats or {}
    name2values = defaultdict(list)
    for single_eval_result in single_eval_results:
        for name, value in single_eval_result.metrics.items():
            name2values[name].append(value)
        if single_eval_result.score is not None:
            name2values["score"].append(single_eval_result.score)
    final_metrics = {}
    for name, values in name2values.items():
        stats = name2stats.get(name, default_stats)
        for stat in stats:
            key = name if stat == "mean" else f"{name}:{stat}"
            final_metrics[key] = _compute_stat(values, stat)
    
    response_data = []
    for single_eval_result in single_eval_results:
        single_eval_result = single_eval_result.__dict__
        if len(single_eval_result) > 2:
            single_eval_result.pop("metrics")
            single_eval_result.pop("score")
            response_data.append(single_eval_result)

    return EvalResult(
        score=final_metrics.pop("score", None), metrics=final_metrics
    )


def process_worker(task_queue, done_queue, worker_func):    
    for line in iter(task_queue.get, "STOP"):
        result = worker_func(line)

        done_queue.put(result)

    done_queue.put("COMPLETE")


def map_with_progress(f: callable, xs: list[Any], num_threads: int = 50):
    num_processes = num_threads
    QUEUE_SIZE = 3000
    
    task_queue, done_queue = Queue(maxsize=QUEUE_SIZE), Queue(maxsize=QUEUE_SIZE)

    def read_data_into_queue():        
        for line in xs:
            task_queue.put(line)

        for _ in range(num_processes):
            task_queue.put('STOP')

    processes = []
    for _ in range(num_processes):
        process = Process(target=process_worker,
                    args=(task_queue, done_queue, f))
        process.start()
        processes.append(process)

    process = Process(target=read_data_into_queue)
    process.start()

    progress_bar = tqdm(total=len(xs))
    num_finished = 0
    
    results = []
    while num_finished < num_processes:
        item = done_queue.get()
        if item == 'COMPLETE':
            num_finished += 1
        else:
            results.append(item)
            progress_bar.update(1)

    progress_bar.close()

    return results


def map_with_ordered_progress(f: callable, xs: list[Any], num_threads: int = 50):
    num_processes = num_threads
    QUEUE_SIZE = 3000
    
    task_queue, done_queue = Queue(maxsize=QUEUE_SIZE), Queue(maxsize=QUEUE_SIZE)

    def read_data_into_queue():        
        for i, line in enumerate(xs):
            task_queue.put((i, line))

        for _ in range(num_processes):
            task_queue.put('STOP')

    processes = []
    for _ in range(num_processes):
        process = Process(target=process_worker,
                    args=(task_queue, done_queue, lambda x: (x[0], f(x[1]))))
        process.start()
        processes.append(process)

    process = Process(target=read_data_into_queue)
    process.start()

    progress_bar = tqdm(total=len(xs))
    num_finished = 0
    
    results_with_index = []
    while num_finished < num_processes:
        item = done_queue.get()
        if item == 'COMPLETE':
            num_finished += 1
        else:
            results_with_index.append(item)
            progress_bar.update(1)

    progress_bar.close()

    # Sort results by index and return only the results without indices
    results_with_index.sort(key=lambda x: x[0])
    results = [r[1] for r in results_with_index]
    
    return results



EXTRACTION_TEMPLATE = """
Look at the following math problem and extract the final answer, such final results or option. If you cannot find an answer, return `No Answer`

## Question: 
{question}

## Answer: 
{answer}

Put the answer in the format of the following example: 

<ANSWER>: <your answer>

Example:
<ANSWER>: A
<ANSWER>: A: 130
<ANSWER>: a = 3
<ANSWER>: 100

If the question is a multiple-choice question, extract the option and value that matches the answer.
"""

CHOICE_EXTRACTION_TEMPLATE = """
We have a multiple-choice question with several choices (in ## Question). We also have an answer to this question (in ## Answer). Please determine which choice is selected based on the answer and the choices in the question.

## Question: 
{question}

## Answer: 
{answer}

Put the selected choice in the format of the following example: 

<CHOICE>: <your answer>

Example:
<CHOICE>: A
<CHOICE>: B
<CHOICE>: C
<CHOICE>: D
"""


def extract_answer(sampler, question, response):
    if response == "":
        return ""
    
    original_response = response
    if "<answer>" in response and "</answer>" in response:
        resp_text = re.findall(r"<answer>([\s\S]+?)</answer>", response)[-1].strip()
    else:
        response = response.strip().split("\n")
        resp_text = [x for x in response if x.strip()]
        resp_text = "\n".join(resp_text[-5:])

    answer = None
    if "\\box" in resp_text or "\\boxed" in resp_text:
        answer = re.findall(r'\\boxed\{([^{}]*(?:\{[^{}]*\})*[^{}]*)\}', resp_text)
        if len(answer) == 0:
            answer = re.findall(r'\\box\{([^{}]*(?:\{[^{}]*\})*[^{}]*)\}', resp_text)


    if answer: 
        answer = answer[0].strip()
    else:
        answer_template = EXTRACTION_TEMPLATE.format(question=question, answer=resp_text)
        for _ in range(6):
            extracted_answer = sampler([dict(content=answer_template, role="user")])
            if extracted_answer is None:
                answer = ""
                continue
            else:
                answer = extracted_answer.replace("<ANSWER>: ", "").strip()
                break
    return answer



def extract_answer_multi_choice(sampler, question, response):
    response = response.strip().split("\n")
    resp_text = [x for x in response if x.strip()]
    resp_text = "\n".join(resp_text[-5:])

    answer = None
    if "\\box" in resp_text or "\\boxed" in resp_text:
        answer = re.findall(r'\\box\{([^{}]*(?:\{[^{}]*\})*[^{}]*)\}', resp_text)
        if len(answer) == 0:
            answer = re.findall(r'\\boxed\{([^{}]*(?:\{[^{}]*\})*[^{}]*)\}', resp_text)

    if answer is not None and len(answer) > 0:
        resp_text = answer[-1].strip()
    
    answer_template = CHOICE_EXTRACTION_TEMPLATE.format(question=question, answer=resp_text)
    for _ in range(6):
        extracted_answer = sampler([dict(content=answer_template, role="user")])
        if extracted_answer is None:
            answer = ""
            continue
        else:
            answer = extracted_answer.replace("<CHOICE>: ", "").strip()
            break
            
    return answer

def compute_repeat_metrics(success, num_repeats, worst_of_n, eval_result):
    """
    计算多次重复评测的指标
    
    Args:
        success: 成功的评测结果列表
        num_repeats: 重复评测次数
        worst_of_n: 是否计算worst-of-n指标
        eval_result: 原始评测结果对象
        
    Returns:
        更新后的eval_result对象
    """
    group_size = len(success) // num_repeats
    accuracy_list = []
    
    for i in range(num_repeats):
        start_idx = i * group_size
        end_idx = (i + 1) * group_size
        group_results = success[start_idx:end_idx]
        group_accuracy = sum(r.score for r in group_results) / len(group_results)
        accuracy_list.append(group_accuracy)
        
    print(f"Accuracy for each repeat: {accuracy_list}")
    
    eval_result.metrics['accuracy_list'] = accuracy_list
    eval_result.metrics['accuracy'] = sum(accuracy_list) / len(accuracy_list)
    eval_result.metrics['accuracy_var'] = sum((x - eval_result.metrics['accuracy']) ** 2 for x in accuracy_list) / len(accuracy_list)
    eval_result.metrics['accuracy_std'] = eval_result.metrics['accuracy_var'] ** 0.5

    if worst_of_n:
        worst_of_n_scores = []
        for i in range(group_size):
            worst_score = 100
            for j in range(num_repeats):
                idx = j * group_size + i
                if success[idx].score < worst_score:
                    worst_score = success[idx].score
            worst_of_n_scores.append(worst_score)
        eval_result.metrics['worst_of_n'] = worst_of_n_scores
        eval_result.metrics['worst_of_n_avg'] = sum(worst_of_n_scores) / len(worst_of_n_scores)
    
    return eval_result