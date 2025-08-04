"""
Measuring Mathematical Problem Solving and Reasoning Ability With the MathBench and ReasoningBench Dataset from Xiaotao Gu's Team.
"""

import random
import re
import os
from pathlib import Path

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
import ast
import time
import inspect
import subprocess
from scicode.parse.parse import read_from_hf_dataset
import shutil


DEFAULT_PROMPT_TEMPLATE = Path("data/scicode/background_comment_template.txt").read_text()
BACKGOUND_PROMPT_TEMPLATE = Path("data/scicode/multistep_template.txt").read_text()


PROB_NUM = 80
DEV_PROB_NUM = 15
STEP_NUM = 288
DEV_STEP_NUM = 50
h5py_file = "./data/scicode/test_data.h5"

def extract_python_script(response: str):
    # We will extract the python script from the response
    if '```' in response:
        python_script = response.split("```python")[1].split("```")[0] if '```python' in response else response.split('```')[1].split('```')[0]
    else:
        print(f"Fail to extract python code from specific format. response: {response}")
        python_script = response
    python_script = re.sub(r'^\s*(import .*|from .*\s+import\s+.*)', '', python_script, flags=re.MULTILINE)
    return python_script


def extract_function_name(function_header):
    pattern = r'\bdef\s+(\w+)\s*\('
    match = re.search(pattern, function_header)
    if match:
        return match.group(1)
    else:
        pattern = r'\bclass\s+(\w+)\s*\('
        match = re.search(pattern, function_header)
        if match:
            return match.group(1)
        else:
            raise ValueError('Function name or class name not found.')

def get_function_from_code(code_string, function_name):
    """
    Extracts and returns the source code of the specified function from a given source code string.

    :param code_string: String containing Python source code
    :param function_name: Name of the function to extract
    :return: String containing the source code of the function, or None if the function is not found
    """
    if code_string is None:
        return None
    try:
        # Parse the code into an AST
        tree = ast.parse(code_string)
        # Iterate through all nodes in the AST
        for node in ast.walk(tree):
            # Check if the node is a function definition
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == function_name:
                # Convert the AST back to a string containing the Python code for the function
                return ast.unparse(node)
    except Exception as e:
        print(f'{function_name} not found with error: {e}')
        return code_string


scicode_tmp_dir = None 


class Gencode:
    def __init__(self, with_background):
        self.model = "default"
        self.output_dir = Path(scicode_tmp_dir, "output")
        self.prompt_dir = Path(scicode_tmp_dir, "prompt")
        self.eval_dir = Path("./data/scicode")
        self.with_background = with_background
        self.previous_llm_code = []

    def _get_background_dir(self):
        return "with_background" if self.with_background else "without_background"

    def save_prompt_with_steps(self, prob_data: dict, prompt: str, num_steps: int) -> None:
        output_dir = self.prompt_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file_path = output_dir / f"{prob_data['problem_id']}.{num_steps}.txt"
        output_file_path.write_text(prompt, encoding="utf-8")

    def save_response_with_steps(self, prob_data: dict, response: str,
                                 previous_code: str, num_steps: int) -> None:
        output_dir = self.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        prob_id = prob_data["problem_id"]
        output_file_path = output_dir / f"{prob_id}.{num_steps}.py"
        python_code = extract_python_script(response)
        output_file_path.write_text(f'{previous_code}\n{python_code}', encoding="utf-8")

    def generate_response_with_steps(
        self, prob_data: dict, num_steps: int, tot_steps: int, model="gpt-4o",
            prompt_template=DEFAULT_PROMPT_TEMPLATE,
            *, save: bool = True, sampler=None) -> None:
        """

        Args:
            prob_data (dict): dict of the problem
            num_steps (int): Current generating step
            tot_steps (int): Total step of the problem
            model (str)
            prompt_template (str)
            save (bool, optional): Save propmt and model response. Defaults to True.
        """
        prob_id = prob_data["problem_id"]
        output_file_path = Path(self.output_dir, f"{prob_id}.{num_steps}.py")
        
        if num_steps == 1:
            self.previous_llm_code = [None] * tot_steps
        else:
            if len(self.previous_llm_code) != tot_steps:
                self.previous_llm_code = [None] * tot_steps
            for prev_step in range(num_steps - 1):
                if self.previous_llm_code[prev_step] is None:
                    if (prob_id == "13" and prev_step == 5) or (prob_id == "62" and prev_step == 0)\
                            or (prob_id == "76" and prev_step == 2):
                        prev_file_path = Path(self.eval_dir, f"{prob_id}.{prev_step + 1}.txt")
                    else:
                        prev_file_path = Path(self.output_dir, f"{prob_id}.{prev_step + 1}.py")
                    if prev_file_path.is_file():
                        prev_file_content = prev_file_path.read_text(encoding='utf-8')
                        func_name = extract_function_name(prob_data["sub_steps"][prev_step]["function_header"])
                        function_code = get_function_from_code(prev_file_content, func_name)
                        self.previous_llm_code[prev_step] = function_code
                    else:
                        raise Exception(f'Generating {prob_id} step {num_steps} ahead of step {prev_step + 1}.')

        if output_file_path.exists():
            return
        prompt, previous_code = self.generate_prompt_with_steps(prob_data, num_steps, prompt_template)
        if save:
            self.save_prompt_with_steps(prob_data, prompt, num_steps)
        
        prompt_messages = [{"role": "user", "content": prompt}]
        #print(f"prompt: {prompt_messages}")
        response_from_llm = sampler(prompt_messages)
        
        self.previous_llm_code[num_steps - 1] = extract_python_script(response_from_llm)
        self.save_response_with_steps(prob_data, response_from_llm, previous_code, num_steps)

    @staticmethod
    def process_problem_code(prob_data: dict, num_steps: int) -> str:
        header_docstring = prob_data['sub_steps'][num_steps - 1]['function_header']
        return_str = prob_data['sub_steps'][num_steps - 1]['return_line']
        string = f"{header_docstring}\n\n{return_str}"
        return string

    def process_problem_steps(self, problem_data: dict, num_steps: int):
        """Process problem data and return previous steps and next steps"""
        output_lines = []
        next_step = []
        previous_code = []
        for i in range(num_steps - 1):
            output_lines.append(problem_data["sub_steps"][i]["step_description_prompt"] + '\n' +
                                problem_data["sub_steps"][i]["step_background"] if self.with_background
                                else problem_data["sub_steps"][i]["step_description_prompt"])
            if not self.previous_llm_code[i]:
                output_lines.append("")
                previous_code.append("")
            else:
                output_lines.append(self.previous_llm_code[i])
                previous_code.append(self.previous_llm_code[i])
            output_lines.append("------")

        next_step.append(problem_data["sub_steps"][num_steps - 1]["step_description_prompt"] + '\n' +
                         problem_data["sub_steps"][num_steps - 1]["step_background"] if self.with_background
                         else problem_data["sub_steps"][num_steps - 1]["step_description_prompt"])
        next_step.append(self.process_problem_code(problem_data, num_steps))
        output_str = "\n\n".join(output_lines[:-1])  # Remove the last "------"
        next_step_str = "\n\n".join(next_step)
        previous_code_str = "\n".join(previous_code)
        return output_str, next_step_str, previous_code_str

    def generate_prompt_with_steps(self, prob_data: dict, num_steps: int,
                                   prompt_template=DEFAULT_PROMPT_TEMPLATE):
        # parse the input file and extract the content
        problem_steps_str, next_step_str, previous_code_str = self.process_problem_steps(prob_data,
                                                                                         num_steps)
        dependencies = prob_data["required_dependencies"]
        assert next_step_str
        return prompt_template.format(
            problem_steps_str=problem_steps_str,
            next_step_str=next_step_str,
            dependencies=dependencies,
        ), f'{dependencies}\n{previous_code_str}\n'


def test_code(scicode_data):
    log_dir = Path(scicode_tmp_dir, "log")
    #scicode_data = read_from_hf_dataset(split)
    scicode_data = [data for data in scicode_data]
    json_dct = {}
    json_idx = {}

    for prob_data in scicode_data:
        json_dct[prob_data['problem_id']] = len(prob_data['sub_steps'])
        json_idx[prob_data['problem_id']] = scicode_data.index(prob_data)
    start_time = time.time()

    code_dir_ = Path(scicode_tmp_dir, "output")
    tmp_dir = Path(scicode_tmp_dir, f'tmp_{start_time}')

    tmp_dir.mkdir(parents=True, exist_ok=True)

    for file_path in code_dir_.iterdir():
        if file_path.is_file() and '.py' in file_path.suffixes:
            file_name = file_path.stem
            file_id = file_name.split(".")[0]
            file_step = file_name.split(".")[1]

            code_content = file_path.read_text(encoding='utf-8')
            json_content = scicode_data[json_idx[file_id]]
            step_id = json_content["sub_steps"][int(file_step) - 1]["step_number"]
            test_lst = json_content["sub_steps"][int(file_step) - 1]["test_cases"]
            assert_file = Path(tmp_dir, f'{step_id}.py')
            with open(assert_file, 'w', encoding='utf-8') as f:
                f.write(code_content)
                f.write(f"""

from scicode.parse.parse import process_hdf5_to_tuple

""")
                f.write(f"targets = process_hdf5_to_tuple('{step_id}', {len(test_lst)}, '{h5py_file}')" + '\n')
                for idx in range(len(test_lst)):
                    f.write(f"target = targets[{idx}]\n\n")
                    for line in test_lst[idx].split('\n'):
                        f.write(line + '\n')

    def run_script(script_path):
        try:
            subprocess.run(['python3', script_path], check=True, capture_output=True,
                           text=True, timeout=60)
            return 0
        except subprocess.CalledProcessError as e:
            print(f"Error running script {script_path}: {e}")
            print(e.output)
            return 1
        except subprocess.TimeoutExpired as e:
            print(f"Runtime error while running script {script_path}: {e}")
            return 2

    correct_prob = np.zeros(PROB_NUM)
    tot_prob = np.zeros(PROB_NUM)
    correct_step = []
    correct_dict = {}

    for i in range(PROB_NUM):
        correct_dict[f'{i+1}'] = []
    
    file_path_list = [x for x in tmp_dir.iterdir()]

    def fn(file_path):
        c_prob = 0
        t_prob = 0
        c_step = None
        c_dict = None
        prob_id = None
        if file_path.is_file():
            func_id = file_path.stem
            prob_id = func_id.split('.')[0]
            #tot_prob[int(prob_id) - 1] += 1
            t_prob += 1
            
            logs_dir_ = Path(log_dir)
            logs_dir_.mkdir(parents=True, exist_ok=True)
            logs_file = Path(logs_dir_, f'{file_path.stem}.txt')
            if logs_file.exists():
                with open(logs_file, 'r') as f:
                    content = f.read().splitlines()
                    if content[0] == 'pass':
                        # correct_prob[int(prob_id) - 1] += 1
                        # correct_step.append(func_id)
                        # correct_dict[prob_id].append(func_id)
                        c_prob += 1
                        c_step = func_id
                        c_dict = func_id
                return (prob_id, t_prob, c_prob, c_step, c_dict)
            ret = run_script(file_path)
            if ret == 0:
                # correct_prob[int(prob_id) - 1] += 1
                # correct_step.append(func_id)
                # correct_dict[str(prob_id)].append(func_id)
                c_prob += 1
                c_step = func_id
                c_dict = func_id
                with open(logs_file, 'w') as f:
                    f.write('pass')
            elif ret == 1:
                with open(logs_file, 'w') as f:
                    f.write('fail')
            else:
                with open(logs_file, 'w') as f:
                    f.write('time out')
        return (prob_id, t_prob, c_prob, c_step, c_dict)

    results = common.map_with_progress(fn, file_path_list, num_threads=10)
    
    sub_correct_prob = 0
    sub_tot_prob = 0
    
    for result_item in results:
        prob_id, t_prob, c_prob, c_step, c_dict = result_item
        if prob_id != None:
            tot_prob[int(prob_id) - 1] += t_prob
            correct_prob[int(prob_id) - 1] += c_prob
            correct_step.append(c_step)
            correct_dict[str(prob_id)] = c_dict
            
            sub_tot_prob += t_prob
            sub_correct_prob += c_prob

    test_time = time.time() - start_time

    correct_prob_num = sum(1 for i in range(PROB_NUM) if
                           correct_prob[i] == tot_prob[i]
                           and tot_prob[i] != 0)
    
    return sub_correct_prob / sub_tot_prob, correct_prob_num / (PROB_NUM - DEV_PROB_NUM)
    
    print(f"correct_prob_num: {correct_prob_num}, {PROB_NUM - DEV_PROB_NUM}")


class SciCodeEval(Eval):
    def __init__(self, num_examples: Optional[int] = None, data_dir: str = "data", proc_num: int = 50, num_repeat: int = 1, model_name: str = "cot", save_dir: str = "", with_background:bool=True):
        examples = self.prepare_dataset(data_dir)
        if num_examples and num_examples >= 0:
            examples = random.Random(0).sample(examples, num_examples)
        if num_repeat > 1:
            examples = examples * num_repeat
            
        self.examples = examples
        self.proc_num = proc_num
        self.dataset_type = "scicode"
        self.num_repeat = num_repeat
        self.model_name = model_name
        self.data_dir = data_dir
        self.save_dir = save_dir
        global scicode_tmp_dir
        scicode_tmp_dir = Path(save_dir, "simple_evals", "scicode")
        self.with_background = with_background

    def prepare_dataset(self, data_dir):
        print(__file__, data_dir)
        examples = [json.loads(line) for line in open(os.path.join(data_dir, "scicode/problems_all.jsonl"))]
        return examples

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        
        def fn(row: dict):
            prob_id = row['problem_id']
            steps = len(row['sub_steps'])
            
            #prompt_template = DEFAULT_PROMPT_TEMPLATE
            prompt_template = BACKGOUND_PROMPT_TEMPLATE if self.with_background else DEFAULT_PROMPT_TEMPLATE
            gcode = Gencode(self.with_background)
            
            for i in range(steps):
                if (prob_id == "13" and i == 5) or (prob_id == "62" and i == 0)\
                        or (prob_id == "76" and i == 2):
                    continue

                gcode.generate_response_with_steps(row, i+1, steps, prompt_template=prompt_template, sampler=sampler)

        num_thread = self.proc_num
        common.map_with_progress(fn, self.examples, num_threads=num_thread)
        
        sub_score, main_score = test_code(self.examples)

        return EvalResult(sub_score, {"main_score": main_score}), None
