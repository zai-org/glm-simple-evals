"""
Measuring Mathematical Problem Solving With the MATH Dataset
Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, Jacob Steinhardt
https://arxiv.org/abs/2103.03874
"""

import random
import os
from functools import partial
import pandas
from typing import Optional

from evals import common
from utils.types import Eval, EvalResult, SamplerBase, SingleEvalResult
from evals.grading.grader import grade_answer


QUERY_TEMPLATE = """
Solve the following math problem step by step. Put your answer inside \\boxed{{}}.

{Question}

Remember to put your answer inside \\boxed{{}}.""".strip()


EQUALITY_TEMPLATE = r"""Look at the following two expressions (answers to a math problem) and judge whether they are equivalent. Only perform trivial simplifications

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

---

YOUR TASK


Respond with only "Yes" or "No" (without quotes). Do not include a rationale.

    Expression 1: %(expression1)s
    Expression 2: %(expression2)s""".strip()


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    except:
        return None


def extract_boxed_answer(solution: str) -> str:
    """Extract the answer from inside a LaTeX \\boxed{} command"""
    solution = last_boxed_only_string(solution)
    solution = remove_boxed(solution)
    return solution


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


class MathEval(Eval):
    def __init__(
        self,
        equality_checker: SamplerBase,
        n_repeats: int = 3,
        num_examples: Optional[int] = None,
        data_dir: str = "data",
        proc_num: int = 50,
        auto_extract_answer=False,
        extractor: SamplerBase = None,
    ):
        df = pandas.read_json(os.path.join(data_dir, "math/MATH500.jsonl"), lines=True)
        examples = [row.to_dict() for _, row in df.iterrows()]
        examples = examples * n_repeats
        if num_examples > 0:
            examples = random.Random(0).sample(examples, num_examples)
        self.examples = examples
        self.equality_checker = equality_checker
        self.proc_num = proc_num
        self.auto_extract_answer = auto_extract_answer
        self.extractor = extractor

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        proc_num = min(self.proc_num, 50)
        results = common.map_with_progress(
            partial(process_func, sampler, self.equality_checker),
            self.examples,
            num_threads=proc_num,
        )

        response_data = [x[1] for x in results]
        results = [x[0] for x in results]
        success = [x for x in results if x is not None]
        failed = [x for x in results if x is None]
        print(f"MATH evaluation: {len(success)} successful, {len(failed)} failed")
        return common.aggregate_results(success), response_data
