"""
Humanity's Last Exam
https://arxiv.org/abs/2501.14249
"""
import random
import re
import os
import json
from typing import Optional
from functools import partial

import math
import numpy as np

from evals import common
from utils.types import Eval, EvalResult, SamplerBase, SingleEvalResult

SYSTEM_PROMPT_FOR_EXACT_MATCH = """
Your response should be in the following format:
Explanation: {your explanation for your final answer}
Exact Answer: {your succinct, final answer}
Confidence: {your confidence score between 0% and 100% for your answer}
""".strip()

SYSTEM_PROMPT_FOR_MULTIPLE_CHOICE = """
Your response should be in the following format:
Explanation: {your explanation for your answer choice}
Answer: {your chosen answer}
Confidence: {your confidence score between 0% and 100% for your answer}
""".strip()

def format_message(question):
    system_prompt = SYSTEM_PROMPT_FOR_EXACT_MATCH if question["answer_type"] == "exactMatch" else SYSTEM_PROMPT_FOR_MULTIPLE_CHOICE
    question_text = question['question']

    messages = [
        {"role": "user", "content": system_prompt + "\n\n" + question_text}
    ]
    return messages

JUDGE_PROMPT = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|\%| and 100|\%| from [response]. Put 100 if there is no confidence score available."""

CORRECTNESS_PATTERN = r"(?i)\*{0,2}correct\*{0,2}\s*:\s*(no|yes)"

CONFIDENCE_PATTERN = r"(?i)\*{0,2}confidence\*{0,2}\s*:\s*([\d\.]+)"

EXTRACTED_PATTERN = r"(?i)\*{0,2}extracted_final_answer\*{0,2}\s*:\s*(.+)"

def format_judge(question, correct_answer, response):
    prompt = JUDGE_PROMPT.format(question=question, correct_answer=correct_answer, response=response)
    messages = [
        dict(content=prompt, role="user")
    ]
    return messages

def process_func(sampler, judge_sampler: SamplerBase, row: dict):
    messages = format_message(row)
    response_text = sampler(messages)
    if len(response_text) > 1024:
        response_text_judge = response_text[-1024:]
    else:
        response_text_judge = response_text
    judge_messages = format_judge(row["question"], row["answer"], response_text_judge)
    trys = 0
    judgement = ""
    correctness, confidence, extracted_final_answer = None, None, None
    while trys < 5:
        judgement = judge_sampler(judge_messages)
        match = re.search(CORRECTNESS_PATTERN, judgement, flags=re.IGNORECASE)
        correctness = match.group(1) if match else None
        match = re.search(CONFIDENCE_PATTERN, judgement, flags=re.IGNORECASE)
        confidence = match.group(1) if match else None
        match = re.search(EXTRACTED_PATTERN, judgement, flags=re.IGNORECASE)
        extracted_final_answer = match.group(1) if match else None
        if extracted_final_answer is None and row["answer_type"] == "multipleChoice":
            choice = random.choice(["A","B","C","D","E"])
            judgement = "Guesswork"
            correctness = "yes" if choice == row["answer"] else "no"
            confidence = 0.2
            extracted_final_answer = choice
            break
        if correctness and extracted_final_answer:
            try:
                correctness = correctness.lower()
                if confidence:
                    confidence = eval(confidence.rstrip("%")) * 0.01
                else:
                    confidence = 1.0
                break
            except:
                pass
        trys += 1
    if correctness is None:
        correctness = "no"
        confidence = 1.0
        extracted_final_answer = extracted_final_answer or None
    accuracy = 1.0 if correctness == "yes" else 0
    
    row["response_text"] = response_text
    row["extracted_final_answer"] = extracted_final_answer
    row["judgement"] = judgement
    row["accuracy"] = accuracy
    row["confidence"] = confidence
    return SingleEvalResult(score=accuracy, metrics={"confidence": confidence}), row

# source: https://github.com/centerforaisafety/hle/blob/main/hle_eval/run_judge_results.py#L97
def calib_err(confidence, correct, p='2', beta=100): 
    # beta is target bin size
    idxs = np.argsort(confidence)
    confidence = confidence[idxs]
    correct = correct[idxs]
    bins = [[i * beta, (i + 1) * beta] for i in range(len(confidence) // beta)]
    bins[-1] = [bins[-1][0], len(confidence)]

    cerr = 0
    total_examples = len(confidence)
    for i in range(len(bins) - 1):
        bin_confidence = confidence[bins[i][0]:bins[i][1]]
        bin_correct = correct[bins[i][0]:bins[i][1]]
        num_examples_in_bin = len(bin_confidence)

        if num_examples_in_bin > 0:
            difference = np.abs(np.nanmean(bin_confidence) - np.nanmean(bin_correct))

            if p == '2':
                cerr += num_examples_in_bin / total_examples * np.square(difference)
            elif p == '1':
                cerr += num_examples_in_bin / total_examples * difference
            elif p == 'infty' or p == 'infinity' or p == 'max':
                cerr = np.maximum(cerr, difference)
            else:
                assert False, "p must be '1', '2', or 'infty'"

    if p == '2':
        cerr = np.sqrt(cerr)

    return cerr

class HLEEval(Eval):
    def __init__(self, equality_checker, num_examples: Optional[int] = None, data_dir: str = "data", proc_num: int = 50, version: int = 2500):
        examples = []
        filename = "hle-3k.jsonl" if version == 3000 else "hle.jsonl"
        dataset = [json.loads(line) for line in open(os.path.join(data_dir, "hle", filename), encoding='utf-8')]
        for row in dataset:
            examples.append({
                'id': row['id'],
                'question': row['question'],
                'answer': row['answer'],
                'answer_type': row['answer_type'],
                'category': row['category']
            })
        if num_examples:
            examples = random.Random(0).sample(examples, num_examples)
        self.examples = examples
        self.proc_num = proc_num
        self.judge_sampler = equality_checker

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        results = common.map_with_progress(partial(process_func, sampler, self.judge_sampler), self.examples, num_threads=self.proc_num)
        results, response_data = [x[0] for x in results], [x[1] for x in results]

        # source: https://github.com/centerforaisafety/hle/blob/main/hle_eval/run_judge_results.py#L148
        correct = np.array([item.score for item in results])
        confidence = np.array([item.metrics["confidence"] for item in results])
        n = len(self.examples)
        accuracy = round(100 * sum(correct) / n, 2)
        # Wald estimator, 95% confidence interval
        if n < 100:
            calibration_error = None
        else:
            calibration_error = 100 * round(calib_err(confidence, correct, p='2', beta=100), 2)
        final_results = common.aggregate_results(results)
        final_results.metrics["calibration_error"] = calibration_error
        return final_results, response_data

