import json
import os

from functools import partial
import random
import string

from evals.aime_eval import AimeEval
from evals.gpqa_eval import GPQAEval
from evals.math_eval import MathEval
from evals.livecodebench_eval import LiveCodeBenchEval
from evals.scicode_eval import SciCodeEval
from evals.mmlu_pro_eval import MMLUProEval
from evals.hle_eval import HLEEval

from samplers.openai_sampler import OpenAISampler
from samplers.zai_sampler import ZaiSampler

import argparse

NO_PROXY = os.environ.get("no_proxy", "")
NO_PROXY = NO_PROXY.split(",") if NO_PROXY else []

def generate_random_string(length=8):
    characters = string.ascii_letters + string.digits
    random_string = ''.join(random.choice(characters) for _ in range(length))
    return random_string


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--zai_api_key', type=str, default="")
    parser.add_argument('--openai_api_key', type=str, default=None)
    parser.add_argument('--openai_base_url', type=str, default=None)
    parser.add_argument("--checker_model_name", type=str, default="Meta-Llama-3.1-70B-Instruct")
    parser.add_argument('--checker_api_key', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--proc_num', type=int, default=5)
    parser.add_argument('--backbone', type=str, default="zai")
    parser.add_argument('--model_name', type=str, default="glm-4.5")
    parser.add_argument('--checker_url', type=str, default="http://172.27.0.46:8000/v1")
    parser.add_argument('--tasks', type=str, nargs="*", default=["aime2024"])
    parser.add_argument('--auto_extract_answer', action="store_true", default=False)
    parser.add_argument('--max_new_tokens', type=int, default=2048)
    parser.add_argument('--max_length', type=int, default=131072)
    parser.add_argument('--temperature', type=float, default=0.6)
    parser.add_argument('--worst_of_n', action="store_true", default=False)
    parser.add_argument('--debug', action="store_true", default=False)
    parser.add_argument('--lcb_date', type=str, default="latest")
    parser.add_argument('--stream', action="store_true", default=False)
    parser.add_argument('--sci_without_background', action="store_true", default=False)
    
    args = parser.parse_args()
    debug = args.debug

    os.makedirs(args.save_dir, exist_ok=True)

    if args.checker_model_name is None:
        args.checker_model_name = generate_random_string(32)
    extractor=None
    if not args.checker_model_name:
        equality_checker = None
    elif args.checker_model_name in ["Meta-Llama-3.1-70B-Instruct"]:
        equality_checker = OpenAISampler(api_key="default", url=args.checker_url, model=args.checker_model_name, temperature=0.0, max_tokens=1024)
    elif args.checker_model_name in ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4-turbo-preview"]:
        equality_checker = OpenAISampler(api_key=args.checker_api_key, url=args.checker_url, model=args.checker_model_name, max_tokens=4096)
    else:
        raise ValueError(f"Unknown equality checker model {args.checker_model_name}")

    if args.backbone == "zai":
        sampler = ZaiSampler(model=args.model_name, api_key=args.zai_api_key, max_tokens=args.max_new_tokens, stream=args.stream)
    elif args.backbone == "openai":
        sampler = OpenAISampler(url=args.openai_base_url, api_key=args.openai_api_key, model=args.model_name, max_tokens=args.max_new_tokens, stream=args.stream)
    else:
        raise ValueError(f"Unknown backbone {args.backbone}")

    eval_dict = {
        'lcb': partial(LiveCodeBenchEval, num_examples=1 if debug else -1, data_dir=args.data_dir, proc_num=args.proc_num, num_repeat=1 if debug else 2, date=args.lcb_date),
        'scicode': partial(SciCodeEval, num_examples=5 if debug else -1, data_dir=args.data_dir, proc_num=args.proc_num, num_repeat=3, save_dir=args.save_dir, with_background=not args.sci_without_background),
        'gpqa': partial(GPQAEval, equality_checker=equality_checker, n_repeats=1 if debug else 10, num_examples=5 if debug else None, data_dir=args.data_dir, proc_num=args.proc_num, auto_extract_answer=args.auto_extract_answer),
        'aime2024': partial(AimeEval, equality_checker=equality_checker, num_examples=1 if debug else -1, year=2024, data_dir=args.data_dir, proc_num=args.proc_num, n_repeats=1 if debug else 32, auto_extract_answer=args.auto_extract_answer, worst_of_n=args.worst_of_n, extractor=extractor),
        'aime2025': partial(AimeEval, equality_checker=equality_checker, num_examples=30 if debug else -1, year=2025, data_dir=args.data_dir, proc_num=args.proc_num, n_repeats=1 if debug else 32, auto_extract_answer=args.auto_extract_answer, worst_of_n=args.worst_of_n, extractor=extractor),
        'math500': partial(MathEval, equality_checker=equality_checker, num_examples=5 if debug else -1, data_dir=args.data_dir, proc_num=args.proc_num, n_repeats=3, extractor=extractor),
        'mmlu_pro': partial(MMLUProEval, num_examples=5 if debug else None, data_dir=args.data_dir, proc_num=args.proc_num),
        'hle': partial(HLEEval, equality_checker=equality_checker, num_examples=5 if debug else None, data_dir=args.data_dir, proc_num=args.proc_num)
    }

    eval_tasks = args.tasks
    if args.tasks[0] == "all":
        eval_tasks = list(eval_dict.keys())
    print(f"### eval_tasks: {eval_tasks}")

    os.makedirs(os.path.join(args.save_dir, 'simple_evals'), exist_ok=True)
    
    for eval_name, eval_obj in eval_dict.items():
        if eval_name not in eval_tasks:
            continue
        print("Evaluating " + eval_name)
        
        result = eval_obj()(sampler)
        if isinstance(result, tuple):
            assert len(result) == 2 
            if len(result) == 2:
                result, response_data = result
        else:
            response_data = None
            
        metrics = result.metrics | {"score": result.score}
        result_filename = os.path.join(args.save_dir, 'simple_evals', f"{eval_name}.json")
        print("metrics: ", metrics)
        with open(result_filename, "w") as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)

        if response_data:
            os.makedirs(os.path.join(args.save_dir, 'simple_evals', eval_name), exist_ok=True)
            with open(os.path.join(args.save_dir, 'simple_evals', eval_name, f"data.jsonl"), "w") as f:
                f.writelines([json.dumps(x, ensure_ascii=False) + "\n" for x in response_data])
    
