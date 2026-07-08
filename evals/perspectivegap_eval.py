import importlib
import json
import random
from functools import partial
from pathlib import Path
from typing import Any, Optional

from evals import common
from utils.types import Eval, EvalResult, SamplerBase, SingleEvalResult

DATASET_ID = "sun1245/PerspectiveGap"
ROLE_ASSIGNMENT = "role_assignment"
PROMPT_WRITING = "prompt_writing"
TASKS = (ROLE_ASSIGNMENT, PROMPT_WRITING)
INSTALL_HINT = (
    "PerspectiveGap scoring is required for this evaluation. Install it with "
    "`pip install 'perspective-gap @ git+https://github.com/WhymustIhaveaname/PerspectiveGap.git@9c6921b3337ff3e6a6a453f68d117a8c1663135e'` "
    "or `uv pip install 'perspective-gap @ git+https://github.com/WhymustIhaveaname/PerspectiveGap.git@9c6921b3337ff3e6a6a453f68d117a8c1663135e'`."
)


def _load_scoring_module():
    try:
        return importlib.import_module("perspective_gap.scoring")
    except ImportError as exc:
        raise ImportError(INSTALL_HINT) from exc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_examples(data_dir: str = "data") -> list[dict[str, Any]]:
    root = Path(data_dir)
    candidates = [
        root if root.is_file() else None,
        root / "perspective_gap" / "evaluations.jsonl",
        root / "evaluations.jsonl",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return _read_jsonl(candidate)

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Install `datasets` or place PerspectiveGap data at "
            "<data_dir>/perspective_gap/evaluations.jsonl."
        ) from exc

    try:
        return list(load_dataset(DATASET_ID, split="test"))
    except Exception as exc:
        raise RuntimeError(
            f"Could not load PerspectiveGap data from {DATASET_ID}. Place the "
            "released JSONL at <data_dir>/perspective_gap/evaluations.jsonl or "
            "pass data_dir as a JSONL file."
        ) from exc


def _prompt_field(task: str) -> str:
    if task == ROLE_ASSIGNMENT:
        return "role_assignment_prompt"
    if task == PROMPT_WRITING:
        return "prompt_writing_prompt"
    raise ValueError(f"Unknown PerspectiveGap task: {task}")


def _score_response(
    task: str,
    response: str,
    row: dict[str, Any],
    scoring: Optional[Any] = None,
) -> dict[str, Any]:
    if scoring is None:
        scoring = _load_scoring_module()
    if task == ROLE_ASSIGNMENT:
        return scoring.score_role_assignment(
            response,
            row["reference_need_sets"],
            row.get("distractor_id"),
        )
    if task == PROMPT_WRITING:
        return scoring.score_prompt_writing(
            response,
            row["fragments"],
            row["reference_need_sets"],
            row.get("distractor_id"),
        )
    raise ValueError(f"Unknown PerspectiveGap task: {task}")


def _failed_score(error: str) -> dict[str, Any]:
    return {
        "pass": False,
        "metrics": {
            "strict_pass": 0.0,
            "net_match_score": 0.0,
            "required_coverage": 0.0,
            "boundary_precision": 0.0,
            "distractor_leakage": 0.0,
        },
        "counts": {"tp": 0, "fp": 0, "fn": 0, "distractor_leak": 0},
        "error": error,
    }


def process_func(sampler: SamplerBase, task: str, row: dict[str, Any]):
    prompt = row[_prompt_field(task)]
    response = ""
    error = None
    try:
        scoring = _load_scoring_module()
        response = sampler([{"content": prompt, "role": "user"}])
        score_result = _score_response(task, response, row, scoring=scoring)
    except ImportError:
        raise
    except Exception as exc:
        error = str(exc)
        score_result = _failed_score(error)

    metrics = score_result.get("metrics") or {}
    counts = score_result.get("counts") or {}
    strict_pass = float(
        metrics.get("strict_pass", float(bool(score_result.get("pass"))))
    )
    score = strict_pass * 100.0
    single_result = SingleEvalResult(score=score, metrics=metrics)
    response_row = {
        "evaluation_id": row.get("evaluation_id"),
        "scenario_id": row.get("scenario_id"),
        "shuffle_seed": row.get("shuffle_seed"),
        "task": task,
        "prompt": prompt,
        "response": response,
        "score": score,
        "metrics": metrics,
        "counts": counts,
    }
    if error:
        response_row["error"] = error
    return single_result, response_row


class PerspectiveGapEval(Eval):
    def __init__(
        self,
        task: str,
        data_dir: str = "data",
        proc_num: int = 50,
        num_examples: Optional[int] = None,
    ):
        if task not in TASKS:
            raise ValueError(f"Unknown PerspectiveGap task: {task}")
        _load_scoring_module()
        examples = load_examples(data_dir)
        if num_examples is not None and num_examples > 0:
            examples = random.Random(0).sample(
                examples, min(len(examples), num_examples)
            )
        self.task = task
        self.examples = examples
        self.proc_num = proc_num

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        if self.proc_num <= 1:
            results = [process_func(sampler, self.task, row) for row in self.examples]
        else:
            results = common.map_with_ordered_progress(
                partial(process_func, sampler, self.task),
                self.examples,
                num_threads=self.proc_num,
            )

        response_data = [item[1] for item in results]
        single_results = [item[0] for item in results]
        print(f"PerspectiveGap {self.task} evaluation: {len(single_results)} examples")
        return common.aggregate_results(single_results), response_data
