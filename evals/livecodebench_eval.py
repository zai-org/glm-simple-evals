"""
Measuring Mathematical Problem Solving and Reasoning Ability With the MathBench and ReasoningBench Dataset from Xiaotao Gu's Team.
"""

import random
import re
import os

import pandas
import json
import multiprocessing
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from evals import common
from utils.types import Eval, EvalResult, SamplerBase, SingleEvalResult
import re
import torch
import numpy as np
import sys
from typing import *
from tqdm.auto import tqdm
from collections import defaultdict, Counter
import functools
from concurrent.futures import as_completed, ProcessPoolExecutor
from utils.testing_utils import run_test
import pickle
import base64
import zlib

SYSTEM_MESSAGE_GENERIC = f"You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program."

FORMATTING_MESSAGE_WITH_STARTER_CODE = "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."

FORMATTING_WITHOUT_STARTER_CODE = "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows."

def get_generic_question_template_answer(row):
    prompt = f"### Question:\n{row['prompt']}\n\n"
    if len(row["starter_code"]):
        prompt += (
            f"### Format: {FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        )
        prompt += f"```python\n{row['starter_code']}\n```\n\n"
    else:
        prompt += f"### Format: {FORMATTING_WITHOUT_STARTER_CODE}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += f"### Answer: (use the provided format with backticks)\n\n"
    return prompt

def get_question_template_answer(row, model_name):
    # 如果 o1 在 model_name 里面，由于 o1 没有 system prompt，所以直接在前面加
    if "o1" in model_name.lower():
        chat_messages = [
            {
                "role": "user",
                "content": SYSTEM_MESSAGE_GENERIC
                + "\n\n"
                + get_generic_question_template_answer(row),
            },
        ]
        return chat_messages
    elif "cot" in model_name.lower():

        # ---------------------------
        # _SYSTEM_MESSAGE_GENERIC = f"You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests."

        prompt = ""
        # prompt = "You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.\n\n"
        # prompt = "Generate a correct Python program that matches the specification and passes all tests.\n\n"
        prompt += f"{row['prompt']}\n\n"
        # prompt = f"{row['prompt']}\n\n"

        if len(row["starter_code"]):
            prompt += f"### Format: {FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
            prompt += f"```python\n{row['starter_code']}\n```\n\n"
        else:
            prompt += f"### Format: {FORMATTING_WITHOUT_STARTER_CODE}\n\n"
            prompt += f"```python\n# YOUR CODE HERE\n```\n\n"

        messages = [
            # {"role": "system", "content": _SYSTEM_MESSAGE_GENERIC},
            {"role": "user", "content": prompt},
        ]
    else:
                # ------ standard implementation ------
        prompt = "You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program.\n\n"

        prompt += f"Question:\n{row['prompt']}\n\n"
        if len(row["starter_code"]):
            prompt += f"{FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
            prompt += f"```python\n{row['starter_code']}\n```\n\n"
        else:
            prompt += f"{FORMATTING_WITHOUT_STARTER_CODE}\n\n"
            prompt += f"```python\n# YOUR CODE HERE\n```\n\n"

        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE_GENERIC},
            {"role": "user", "content": prompt},
        ]
        
    return messages

def estimate_pass_at_k(num_samples, num_correct, k):
    """Estimates pass@k of each problem and returns them in an array."""

    def estimator(n: int, c: int, k: int) -> float:
        """Calculates 1 - comb(n - c, k) / comb(n, k)."""
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    import itertools

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array(
        [estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)]
    )


def compute_metrics_from_results(results, k_list=[1, 5]):
    total = []
    correct = []
    task_ids = []
    for task_id, res in enumerate(results):
        all_correct = []
        for generation in res:
            gen = np.array(generation)
            all_correct.append(np.all(gen > 0))
        task_ids.append(task_id)
        total.append(len(all_correct))
        correct.append(sum(all_correct))
    total = np.array(total)
    correct = np.array(correct)
    ks = k_list
    score = correct / total * 100
    # detail_pass_at_k = {
    #     f"pass@{k}": estimate_pass_at_k(total, correct, k).tolist()
    #     for k in ks
    #     if (total >= k).all()
    # }
    return SingleEvalResult(
        score=score,
        metrics={
            f"pass@{k}": estimate_pass_at_k(total, correct, k) * 100
            # this will be aggrated so no need of .mean()
            for k in ks
            if (total >= k).all()
        },
    )


def check_correctness(sample, generation, timeout, debug=True):
    """Check correctness of code generation with a global timeout.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside `run_test`"""

    def _temp_run(sample, generation, debug, result):
        result.append(run_test(sample, test=generation, debug=debug, timeout=timeout))

    manager = multiprocessing.Manager()
    result = manager.list()
    p = multiprocessing.Process(
        target=_temp_run, args=(sample, generation, debug, result)
    )
    p.start()
    #p.join(
    #    timeout=(timeout + 1) * len(json.loads(sample["input_output"])["inputs"]) + 5
    #)
    p.join(
        timeout=(timeout)
    )
 
    if p.is_alive():
        p.kill()
    if not result:
        in_outs = json.loads(sample["input_output"])
        # consider that all tests failed
        result = [[-1 for i in range(len(in_outs["inputs"]))]]
        if debug:
            print(f"global timeout")
    return result[0]


def evaluate_generations_by_problem(args):
    problem_generations: list[str] = args[0]
    sample = args[1]
    debug: bool = args[2]
    timeout: int = args[3]
    res = []
    for o_idx, o in enumerate(problem_generations):
        curr_res = [-2]
        try:
            curr_res = check_correctness(sample, o, timeout=timeout, debug=debug)
            if debug:
                print(f"\nSuccessful compilation of task {o_idx}!")
            fixed = []
            for e in curr_res:
                if isinstance(e, np.ndarray):
                    e = e.item(0)
                if isinstance(e, np.bool_):
                    e = bool(e)
                fixed.append(e)
            curr_res = fixed
            if not np.all(curr_res):
                if debug:
                    print(f"Results were not True for all test cases {curr_res=}\n")
        except Exception as e:
            if debug:
                print(f"Compilation failed, test framework exception = {repr(e)}{e}\n")
            # break
        finally:
            assert isinstance(curr_res, list)
            res.append(curr_res)
    if debug:
        for i, r in enumerate(problem_generations):
            print("Sample\n")
            print(r)
            print("\n")
            print("Result\n")
            print(res[i])
            print("*" * 31 + "\n\n")
    res = compute_metrics_from_results([res], k_list=[1])
    return res

def postprocess_generation(code, sample, dataset_type="humanevalx", mode="instruction"):
    if "###Response" in code:
        code = code.split("###Response")[1]
    if "<think>" in code and "</think>" in code:
        code = code.split("</think>")[1]

    if "```" in code:
        pattern = r'```(.*?)\n(.*?)```'
        matches = re.findall(pattern, code, re.DOTALL)
        if len(matches) == 0:
            print("failed to find python code")
            return None
        
        for match in matches[::-1]:
            if "python" in match[0].lower():
                code = match[1]
                break
            else:
                # old implementation
                code = match[1]
                break
    code = code.replace('\t', '    ')
    code = code.rstrip()
    return code


class LiveCodeBenchEval(Eval):
    def __init__(self, num_examples: Optional[int] = None, data_dir: str = "data", proc_num: int = 50, num_repeat: int = 1, model_name: str = "cot", date="latest"):
        # df = pandas.read_csv(
        #     os.path.join(data_dir, "math/math_test.csv")
        # )
        # examples = [row.to_dict() for _, row in df.iterrows()]

        examples = self.prepare_dataset(data_dir, date)
        if num_examples and num_examples >= 0:
            examples = random.Random(0).sample(examples, num_examples)
        if num_repeat > 1:
            examples = examples * num_repeat
            
        self.examples = examples
        self.proc_num = proc_num
        self.dataset_type = "livecodebench"
        self.num_repeat = num_repeat
        self.model_name = model_name

    def prepare_dataset(self, data_dir, date):
        print(__file__, data_dir)
        # data_dir = "/workspace/qianyi/glm-evals-datasets/data"

        if date == "latest":
            examples = [json.loads(line) for line in open(os.path.join(data_dir, "livecodebench/livecodebench.jsonl"))]
        else:
            if not os.path.exists(os.path.join(data_dir, f"livecodebench/livecodebench_{date}.jsonl")):
                print(f"livecodebench_{date}.jsonl does not exist")
                print("Please input date in the format of `2408_2501`")
                raise ValueError(f"livecodebench_{date}.jsonl does not exist")
            
            examples = [json.loads(line) for line in open(os.path.join(data_dir, f"livecodebench/livecodebench_{date}.jsonl"))]
        examples = [
            {
                "prompt": x["question_content"], 
                # "reference": x["reference"][0], 
                "question_id": x["question_id"],
                "public_test_cases": x["public_test_cases"],
                "metadata": x["metadata"],
                "private_test_cases": x["private_test_cases"],
                "starter_code": x["starter_code"]
            } for x in examples]
        return examples

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        
        def fn(row: dict):
            prompt = row["prompt"]
            # if len(row["starter_code"]):
            #     prompt = "Question:\n" + prompt + f"\nFormat: You will use the following starter code to write the solution to the problem and enclose your code within delimiters.\n```python\n{row['starter_code']}\n```\n\nAnswer: (use the provided format with backticks)\n\n"
            # else:
            #     prompt = "Question:\n" + prompt + f"\nFormat: You will use the following starter code to write the solution to the problem and enclose your code within delimiters.\n```python\n# YOUR CODE HERE\n```\n\nAnswer: (use the provided format with backticks)\n\n"
            # prompt_messages = [dict(content=prompt, role="user")]
            prompt_messages = get_question_template_answer(row, self.model_name)

            # response_text = sampler(prompt_messages, top_p=0.95, temperature=1.00)
            response_text = sampler(prompt_messages)

            #print(response_text)
            sample = row
            prediction = response_text
            
            task_id = sample["question_id"]
            sample["prompt"] = ""
            
            sample["generation"] = postprocess_generation(prediction, sample, dataset_type=self.dataset_type, mode="instruction")

            if sample["generation"] is None or len(sample["generation"]) == 0:
                print(f"generation is None or len(sample['generation']) == 0")
                sample["generation"] = 'return 0'

            test_cases, tmp_inputs, tmp_outputs, tmp_fn = {}, [], [], None
            try:
                private_test_cases = json.loads(sample["private_test_cases"])
            except:
                private_test_cases = json.loads(
                    pickle.loads(
                        zlib.decompress(
                            base64.b64decode(sample["private_test_cases"].encode("utf-8"))
                        )
                    )
                )
            for case in json.loads(sample["public_test_cases"]) + private_test_cases:
                tmp_inputs.append(case["input"])
                tmp_outputs.append(case["output"])
            tmp_fn = json.loads(sample["metadata"]).get("func_name", None)

            test_cases = {"inputs":tmp_inputs,"outputs":tmp_outputs,"fn_name":tmp_fn}
            test_cases = {"input_output":json.dumps(test_cases)}

            timeout = 30
            args = ([sample["generation"]], test_cases, False, timeout)
            result = evaluate_generations_by_problem(args)
            return result, dict(response=response_text, question=prompt, score=result.score.item())

        num_thread = self.proc_num
        results = common.map_with_progress(fn, self.examples, num_threads=num_thread)
        response_data = [x[1] for x in results if x and isinstance(x, tuple) and len(x) > 1]
        results = [x[0] for x in results if x is not None]
        print('LCB validation results num:', len(results))

        return common.aggregate_results(results), response_data
