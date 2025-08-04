"""
Measuring Mathematical Problem Solving With the MATH Dataset
Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, Jacob Steinhardt
https://arxiv.org/abs/2103.03874
"""

import json
import random
import re
import signal
import os
import pandas

from functools import partial

from typing import Optional

from evals import common
from evals.common import extract_answer, MATH_QUERY_TEMPLATE as QUERY_TEMPLATE, ANSWER_PATTERN, check_equality

from utils.types import Eval, EvalResult, SamplerBase, SingleEvalResult
from evals.deepscaler_rule_rm import send_deepscaler_rule_rm_request


def timeout_handler():
    raise TimeoutError(f"Function execution timed out")


def send_deepscaler_rule_rm_request_with_timeout(label, answer, timeout=10):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)  
    
    try:
        result = send_deepscaler_rule_rm_request(label, answer, None)
        signal.alarm(0)
        return result
    except TimeoutError:
        print(f"Function execution timed out after {timeout} seconds")
        return 0
    finally:
        signal.alarm(0) 

def process_func(sampler, equality_checker, auto_extract_answer, extractor, row: dict):
    if auto_extract_answer:
        prompt_messages = [dict(content=row["Question"], role="user")]
        response_text = sampler(prompt_messages)
        if len(response_text) > 1024:
            response_text = response_text[-1024:]
        if extractor:
            extracted_answer = extract_answer(extractor, row["Question"], response_text)
        else:
            extracted_answer = extract_answer(equality_checker, row["Question"], response_text)
    else:
        prompt_messages = [dict(content=QUERY_TEMPLATE.format(**row), role="user")]
        response_text = sampler(prompt_messages)
        if len(response_text) > 1024:
            response_text = response_text[-1024:]
        match = re.search(ANSWER_PATTERN, response_text)
        extracted_answer = match.group(1) if match else None

    if extracted_answer is None:
        extracted_answer = ""
        score = 0
    else:
        rule_based_score = send_deepscaler_rule_rm_request_with_timeout(row["Answer"], extracted_answer, timeout=10)
        if rule_based_score == -1:
            score = check_equality(equality_checker, row["Answer"], extracted_answer, question=row["Question"], qwq_check=True if extractor else False)
        else:
            score = rule_based_score

        if score is None:
            # print("failed to check equality in MATH")
            # return None
            score = 0
    
    score = float(score)
    score = score * 100

    return SingleEvalResult(score=score), dict(problem=row["Question"],   response=response_text, extracted_answer=extracted_answer, label=row["Answer"], score=score)


class AimeEval(Eval):
    def __init__(
        self, 
        equality_checker: SamplerBase, 
        num_examples: Optional[int] = None, 
        year=None, 
        data_dir: str = "data", 
        proc_num: int = 50,
        n_repeats: int = 4,
        auto_extract_answer: bool = False,
        worst_of_n: bool = False,
        extractor: SamplerBase = None
):
        if year == 2025:
            examples = [json.loads(x) for x in open(os.path.join(data_dir, "aime/aime_2025.jsonl"))]
        elif year == "beyond_aime":
            examples = [json.loads(x) for x in open(os.path.join(data_dir, "aime/beyond_aime.jsonl"))]
        else:
            df = pandas.read_csv(
                os.path.join(data_dir, "aime/AIME_Dataset_1983_2024.csv")
            )
            if year is not None:
                examples = [row.to_dict() for _, row in df.iterrows() if row.to_dict()['Year'] == year]
            else:
                examples = [row.to_dict() for _, row in df.iterrows()]

        if num_examples > 0:
            assert n_repeats == 1
            examples = random.Random(0).sample(examples, min(len(examples), num_examples))

        self.examples = examples * n_repeats
        self.equality_checker = equality_checker
        self.proc_num = proc_num
        self.n_repeats = n_repeats
        self.auto_extract_answer = auto_extract_answer
        self.worst_of_n = worst_of_n
        self.extractor = extractor
        
    def __call__(self, sampler: SamplerBase) -> EvalResult:
        results = common.map_with_ordered_progress(partial(process_func, sampler, self.equality_checker, self.auto_extract_answer, self.extractor), self.examples, num_threads=self.proc_num)

        response_data = [x[1] for x in results]
        results = [x[0] for x in results]
        success = [x for x in results if x is not None]
        failed = [x for x in results if x is None]
        print(f"AIME evaluation: {len(success)} successful, {len(failed)} failed")
        
        eval_result = common.aggregate_results(success)
        
        if self.n_repeats > 1:
            eval_result = common.compute_repeat_metrics(
                success=success,
                num_repeats=self.n_repeats,
                worst_of_n=self.worst_of_n,
                eval_result=eval_result
            )
        
        return eval_result, response_data
