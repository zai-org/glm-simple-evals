"""
American Invitational Mathematics Examination
"""

import json
import os
import pandas
import random

from functools import partial
from typing import Optional

from evals import common
from evals.math_eval import (
    QUERY_TEMPLATE,
    EQUALITY_TEMPLATE,
    check_equality,
    extract_boxed_answer,
)
from evals.grading.grader import grade_answer
from utils.types import Eval, EvalResult, SamplerBase, SingleEvalResult


def check_equality(sampler: SamplerBase, expr1: str, expr2: str):
    prompt = EQUALITY_TEMPLATE % {"expression1": expr1, "expression2": expr2}

    for _ in range(3):
        response = sampler([dict(content=prompt, role="user")])
        if len(response) == 0:
            continue
        else:
            break
    if len(response) == 0:
        return 0
    return "yes" in response.lower().strip()


def process_func(sampler, equality_checker, row: dict):
    prompt_messages = [
        dict(content=QUERY_TEMPLATE.format(Question=row["Question"]), role="user")
    ]
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

    score = float(score)
    score = score * 100
    return SingleEvalResult(score=score), dict(
        problem=row["Question"],
        response=response,
        extracted_answer=extracted_answer,
        label=row["Answer"],
        score=score,
    )


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
        extractor: SamplerBase = None,
    ):
        if year == 2025:
            examples = [
                json.loads(x)
                for x in open(os.path.join(data_dir, "aime/aime_2025.jsonl"))
            ]
        else:
            df = pandas.read_csv(
                os.path.join(data_dir, "aime/AIME_Dataset_1983_2024.csv")
            )
            if year is not None:
                examples = [
                    row.to_dict()
                    for _, row in df.iterrows()
                    if row.to_dict()["Year"] == year
                ]
            else:
                examples = [row.to_dict() for _, row in df.iterrows()]

        if num_examples > 0:
            assert n_repeats == 1
            examples = random.Random(0).sample(
                examples, min(len(examples), num_examples)
            )

        self.examples = examples * n_repeats
        self.equality_checker = equality_checker
        self.proc_num = proc_num
        self.n_repeats = n_repeats
        self.auto_extract_answer = auto_extract_answer
        self.worst_of_n = worst_of_n
        self.extractor = extractor

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        results = common.map_with_ordered_progress(
            partial(process_func, sampler, self.equality_checker),
            self.examples,
            num_threads=self.proc_num,
        )

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
                eval_result=eval_result,
            )

        return eval_result, response_data
