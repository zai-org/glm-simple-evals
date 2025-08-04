"""
Measuring Mathematical Problem Solving With the MATH Dataset
Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, Jacob Steinhardt
https://arxiv.org/abs/2103.03874
"""

import json
import random
import re
import os
from functools import partial
import pandas
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from evals import common
from evals.math_eval import check_equality, extract_boxed_answer, grade_answer,QUERY_TEMPLATE
from utils.types import Eval, EvalResult, SamplerBase, SingleEvalResult
from evals.common import extract_answer

def critic_analysis(_response):
    if type(_response) is dict:
        match = None
    else:
        _response_clearn = _response.replace('```', '').strip()
        patterns = [
            r'【判断结果】[\s：:]*(正确|错误)',
            r'【Judgment Result】[\s：:]*(Correct|Incorrect)'
        ]
        for pattern in patterns:
            incorrect_pattern = re.compile(pattern)
            match = incorrect_pattern.search(_response_clearn, re.S)
            if match:
                break
    # If match is found
    if match:
        matched_label = match.group(1)
        if matched_label == "错误" or matched_label == "Incorrect":
            return -1
        elif matched_label == "正确" or matched_label == "Correct":
            return 1
        else:
            return 0
    return 0


def process_func(sampler, equality_checker, k: int, row: dict):
    is_correct = False
        
    save_pass_k = row.copy()
    save_pass_k['k_samples'] = []

    for attempt in range(k):
        prompt_messages = [dict(content=QUERY_TEMPLATE.format(Question=row["Question"]), role="user")]
        response = sampler(prompt_messages)
        extracted_answer = extract_boxed_answer(response)
        
        if extracted_answer is None:
            extracted_answer = ""
            score = 0
        else:
            if grade_answer(extracted_answer, row["Answer"]):
                score = 1
            else:
                score = check_equality(equality_checker, row["Answer"], extracted_answer)

        is_correct = bool(score)
        record = {
            "response": response,
            "concise_ans": extracted_answer,
            "ground_truth_ans": row["Answer"],
            "correctness": bool(score)
        }
        save_pass_k['k_samples'].append(record)

    score = 100.0 if is_correct else 0.0
    return SingleEvalResult(score=score), dict(problem=row["Question"], response=response, extracted_answer=extracted_answer, label=row["Answer"], score=score)


class MathPassKEval(Eval):
    def __init__(self, equality_checker: SamplerBase, num_examples: Optional[int] = None, data_dir: str = "data", proc_num: int = 50, pass_k=1, num_repeats: int = 1, worst_of_n: bool = False, extractor: SamplerBase = None):
        df = pandas.read_json(
            os.path.join(data_dir, "math/MATH500.jsonl"), lines=True
        )
        examples = [row.to_dict() for _, row in df.iterrows()]
        if num_examples > 0:
            examples = random.Random(0).sample(examples, num_examples)
        
        if num_repeats > 1:
            examples = examples * num_repeats
        self.num_repeats = num_repeats
            
        self.examples = examples
        self.equality_checker = equality_checker
        self.proc_num = proc_num
        self.pass_k = pass_k
        self.worst_of_n = worst_of_n
        self.extractor = extractor
        
    def __call__(self, sampler: SamplerBase) -> EvalResult:
        proc_num = min(self.proc_num, 50)
        results = common.map_with_ordered_progress(
            partial(
                process_func, 
                sampler, 
                self.equality_checker, 
                self.pass_k, 
            ), 
            self.examples, num_threads=proc_num
        )
        response_data = [x[1] for x in results]
        results = [x[0] for x in results]
        
        success = [x for x in results if x is not None]
        failed = [x for x in results if x is None]
        print(
            f"MATH evaluation: {len(success)} successful, {len(failed)} failed"
        )
        
        eval_result = common.aggregate_results(success)
        
        if self.num_repeats > 1:
            eval_result = common.compute_repeat_metrics(
                success=success,
                num_repeats=self.num_repeats,
                worst_of_n=self.worst_of_n,
                eval_result=eval_result
            )
        
        return eval_result, response_data
