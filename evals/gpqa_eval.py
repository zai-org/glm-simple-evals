"""
GPQA: A Graduate-Level Google-Proof Q&A Benchmark
David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, Samuel R. Bowman
https://arxiv.org/abs/2311.12022
"""

import random
import re
import os

import pandas
from typing import Optional
from functools import partial
from evals import common
from utils.types import Eval, EvalResult, SamplerBase, SingleEvalResult


QUERY_TEMPLATE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

ANSWER_PATTERN = r"(?i)ANSWER\s*:\s*([A-D])"

QUERY_TEMPLATE_RAW = """{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

def format_question(row, auto_extract_answer=False):
    if not auto_extract_answer:
        return QUERY_TEMPLATE.format(**row)
    else:
        return QUERY_TEMPLATE_RAW.format(**row)

def process_func(sampler, equality_checker, auto_extract_answer, extractor, row: dict):
    choices = [
        row["Correct Answer"],
        row["Incorrect Answer 1"],
        row["Incorrect Answer 2"],
        row["Incorrect Answer 3"],
    ]
    choices = [choices[i] for i in row["permutation"]]
    correct_index = choices.index(row["Correct Answer"])
    correct_answer = "ABCD"[correct_index]
    choices_dict = dict(
        A=choices[0], B=choices[1], C=choices[2], D=choices[3], Question=row["Question"]
    )

    prompt_messages = [dict(content=format_question(choices_dict, auto_extract_answer), role="user")]
    response_text = sampler(prompt_messages)
    if len(response_text) > 1024:
        response_text = response_text[-1024:]
    match = re.search(ANSWER_PATTERN, response_text)
    extract_answer_origin = None
    if auto_extract_answer:
        if extractor:
            extracted_answer = common.extract_answer_multi_choice(extractor, format_question(choices_dict, auto_extract_answer), response_text)
        else:
            extracted_answer = common.extract_answer_multi_choice(equality_checker, format_question(choices_dict, auto_extract_answer), response_text)
        extract_answer_origin = extracted_answer
        if len(extracted_answer.strip()) > 1:
            extracted_answer = extracted_answer.strip()[0]
        if extracted_answer not in "ABCD":
            extracted_answer = random.choice("ABCD")
        score = 1.0 if extracted_answer == correct_answer else 0.0
    else:
        extracted_answer = match.group(1) if match else None
        score = 1.0 if extracted_answer == correct_answer else 0.0
    score = score * 100

    return SingleEvalResult(
        score=score, metrics={"chars": len(response_text), row["High-level domain"]: score}
    ), dict(response=response_text, question=format_question(choices_dict, auto_extract_answer), extract_answer=extract_answer_origin, score=score, correct_answer=correct_answer, domain=row["High-level domain"])

class GPQAEval(Eval):
    def __init__(
        self,
        equality_checker: SamplerBase, 
        n_repeats: int = 4,
        variant: str = "diamond",
        num_examples: Optional[int] = None,  # restrict to a subset of the data for debugging
        data_dir: str = "data",
        proc_num: int = 50,
        auto_extract_answer: bool = False,
        extractor: SamplerBase = None
    ):
        df = pandas.read_csv(
            os.path.join(data_dir, f"gpqa/gpqa_{variant}.csv")
        )
        examples = [row.to_dict() for _, row in df.iterrows()]
        rng = random.Random(0)
        if num_examples:
            assert n_repeats == 1, "n_repeats only supported for num_examples"
            examples = rng.sample(examples, num_examples)
        examples = examples * n_repeats
        examples = [example | {"permutation": rng.sample(range(4), 4)} for example in examples]
        self.examples = examples
        self.n_repeats = n_repeats
        self.proc_num = proc_num
        self.equality_checker = equality_checker
        self.auto_extract_answer = auto_extract_answer
        self.extractor = extractor

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        results = common.map_with_progress(partial(process_func, sampler, self.equality_checker, self.auto_extract_answer, self.extractor), self.examples, num_threads=self.proc_num)
        results_origin = [x[0] for x in results]
        response_data = [x[1] for x in results]
        return common.aggregate_results(results_origin), response_data