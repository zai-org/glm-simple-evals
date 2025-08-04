"""
Measuring Massive Multitask Language Understanding
Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, Jacob Steinhardt
https://arxiv.org/abs/2009.03300
"""

import random
import re
import os
import pandas
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from functools import partial
from datasets import load_dataset

from evals import common
from utils.types import Eval, EvalResult, SamplerBase, SingleEvalResult

def load_mmlu_pro(data_dir):
    dataset = load_dataset(os.path.join(data_dir, 'mmlu_pro'))
    test_df, val_df = dataset["test"], dataset["validation"]
    test_df = preprocess(test_df)
    val_df = preprocess(val_df, group=True)
    return test_df, val_df

def preprocess(test_df, group=False):
    res_df = []
    for each in test_df:
        # print(each)
        options = []
        for opt in each["options"]:
            if opt == "N/A":
                continue
            options.append(opt)
        each["options"] = options
        res_df.append(each)
    if not group:
        return res_df
    res = {}
    for each in res_df:
        if each["category"] not in res:
            res[each["category"]] = []
        res[each["category"]].append(each)
    return res

def format_example(question, options, cot_content=""):
    if cot_content == "":
        cot_content = "Let's think step by step."
    if cot_content.startswith("A: "):
        cot_content = cot_content[3:]
    example = "Question: {}\nOptions: ".format(question)
    choice_map = "ABCDEFGHIJ"
    for i, opt in enumerate(options):
        example += "{}. {}\n".format(choice_map[i], opt)
    # if cot_content == "":
    #     example += "Answer: "
    # else:
    #     example += "Answer: " + cot_content + "\n\n"
    return example


def extract_answer(text):
    pattern = r"answer is \(?([A-J])\)?"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        print("1st answer extract failed\n")
        return extract_again(text)


def extract_again(text):
    match = re.search(r'.*[aA]nswer:\s*([A-J])', text)
    if match:
        return match.group(1)
    else:
        return extract_final(text)


def extract_final(text):
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return None


def single_request(sampler, single_question, cot_examples_dict):
    category = single_question["category"]
    cot_examples = cot_examples_dict[category]
    question = single_question["question"]
    options = single_question["options"]
    prompt = "The following are multiple choice questions (with answers) about {}. Think step by" \
             " step and then output the answer in the format of \"The answer is (X)\" at the end.\n\n" \
        .format(category)
    # for each in cot_examples:
    #     prompt += format_example(each["question"], each["options"], each["cot_content"])
    input_text = format_example(question, options)
    try:
        prompt_messages = [dict(content=prompt + input_text, role="user")]
        response = sampler(prompt_messages)
        response = response.replace('**', '')
    except Exception as e:
        print("error", e)
        return None, None
    pred = extract_answer(response)
    return prompt+input_text, pred, response

def process_func(sampler, cot_examples_dict: dict, row: dict):
    prompt, extracted_answer, response_text = single_request(sampler, row, cot_examples_dict)
    score = 1.0 if extracted_answer == row["answer"] else 0.0
    category = row["category"]
    score = score * 100
    return SingleEvalResult(score=score, metrics={category: score}), dict(
        prompt=prompt,
        question=row['question'],
        category=category,
        response=response_text,
        extracted_answer=extracted_answer,
        correct_answer=row["answer"],
        score=score,
    )


class MMLUProEval(Eval):
    def __init__(self, num_examples: Optional[int] = None, data_dir: str = "data", proc_num: int = 50):
        examples, self.cot_examples_dict = load_mmlu_pro(data_dir)
        if num_examples:
            examples = random.Random(0).sample(examples, num_examples)
        self.examples = examples
        self.proc_num = proc_num

    def __call__(self, sampler: SamplerBase) -> EvalResult:
    
        results = common.map_with_progress(partial(process_func, sampler, self.cot_examples_dict), self.examples, num_threads=self.proc_num)
        results, response_data = [x[0] for x in results], [x[1] for x in results]
        return common.aggregate_results(results), response_data

