import importlib
import json
import sys
import types
from pathlib import Path

import pytest


ROLE_TASK = "role_assignment"
PROMPT_TASK = "prompt_writing"


def import_perspectivegap_module():
    return importlib.import_module("evals.perspectivegap_eval")


def sample_record():
    return {
        "evaluation_id": "pg_unit__seed_1",
        "scenario_id": "pg_unit",
        "shuffle_seed": 1,
        "roles": ["coder", "reviewer"],
        "fragments": [
            {"id": "f1", "text": "Coder-only instructions.", "is_distractor": False},
            {"id": "f2", "text": "Reviewer-only instructions.", "is_distractor": False},
            {"id": "f3", "text": "Distractor instructions.", "is_distractor": True},
        ],
        "distractor_id": "f3",
        "reference_need_sets": {"coder": ["f1"], "reviewer": ["f2"]},
        "role_assignment_prompt": "Return JSON fragment IDs per role.",
        "prompt_writing_prompt": "Write one markdown prompt per role.",
    }


def write_dataset(tmp_path):
    data_dir = tmp_path / "data"
    perspective_gap_dir = data_dir / "perspective_gap"
    perspective_gap_dir.mkdir(parents=True)
    (perspective_gap_dir / "evaluations.jsonl").write_text(
        json.dumps(sample_record()) + "\n"
    )
    return data_dir


def test_load_examples_reports_hf_failure_with_local_path_hint(tmp_path, monkeypatch):
    module = import_perspectivegap_module()
    fake_datasets = types.ModuleType("datasets")

    def load_dataset(*args, **kwargs):
        raise RuntimeError("hub unavailable")

    fake_datasets.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    with pytest.raises(RuntimeError, match="PerspectiveGap data"):
        module.load_examples(str(tmp_path / "missing_data"))


def install_fake_scoring(monkeypatch, calls):
    package = types.ModuleType("perspective_gap")
    scoring = types.ModuleType("perspective_gap.scoring")

    def score_role_assignment(response, reference_need_sets, distractor_id=None):
        calls.append(("role_assignment", response, reference_need_sets, distractor_id))
        return {
            "pass": True,
            "metrics": {"strict_pass": 1.0, "net_match_score": 1.0},
            "counts": {"tp": 2, "fp": 0, "fn": 0, "distractor_leak": 0},
        }

    def score_prompt_writing(
        response, fragments, reference_need_sets, distractor_id=None
    ):
        calls.append(
            ("prompt_writing", response, fragments, reference_need_sets, distractor_id)
        )
        return {
            "pass": False,
            "metrics": {"strict_pass": 0.0, "net_match_score": 0.5},
            "counts": {"tp": 1, "fp": 0, "fn": 1, "distractor_leak": 0},
        }

    scoring.score_role_assignment = score_role_assignment
    scoring.score_prompt_writing = score_prompt_writing
    package.scoring = scoring
    monkeypatch.setitem(sys.modules, "perspective_gap", package)
    monkeypatch.setitem(sys.modules, "perspective_gap.scoring", scoring)


class RecordingSampler:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def __call__(self, message_list):
        self.calls.append(message_list)
        return self.response


def test_perspectivegap_module_imports_without_importing_scorer(monkeypatch):
    monkeypatch.delitem(sys.modules, "perspective_gap.scoring", raising=False)

    module = import_perspectivegap_module()

    assert module.ROLE_ASSIGNMENT == ROLE_TASK
    assert module.PROMPT_WRITING == PROMPT_TASK
    assert "perspective_gap.scoring" not in sys.modules


def test_role_assignment_eval_uses_role_prompt_and_scorer(tmp_path, monkeypatch):
    calls = []
    install_fake_scoring(monkeypatch, calls)
    data_dir = write_dataset(tmp_path)
    module = import_perspectivegap_module()
    sampler = RecordingSampler('{"coder": ["f1"], "reviewer": ["f2"]}')

    eval_result, response_data = module.PerspectiveGapEval(
        task=ROLE_TASK,
        data_dir=str(data_dir),
        num_examples=1,
        proc_num=1,
    )(sampler)

    assert sampler.calls == [
        [{"content": "Return JSON fragment IDs per role.", "role": "user"}]
    ]
    assert calls == [
        (
            "role_assignment",
            '{"coder": ["f1"], "reviewer": ["f2"]}',
            {"coder": ["f1"], "reviewer": ["f2"]},
            "f3",
        )
    ]
    assert eval_result.score == 100.0
    assert response_data[0]["task"] == ROLE_TASK
    assert response_data[0]["evaluation_id"] == "pg_unit__seed_1"


def test_prompt_writing_eval_uses_prompt_writing_prompt_and_scorer(
    tmp_path, monkeypatch
):
    calls = []
    install_fake_scoring(monkeypatch, calls)
    data_dir = write_dataset(tmp_path)
    module = import_perspectivegap_module()
    response = (
        "# coder\nCoder-only instructions.\n# reviewer\nReviewer-only instructions."
    )
    sampler = RecordingSampler(response)

    eval_result, response_data = module.PerspectiveGapEval(
        task=PROMPT_TASK,
        data_dir=str(data_dir),
        num_examples=1,
        proc_num=1,
    )(sampler)

    assert sampler.calls == [
        [{"content": "Write one markdown prompt per role.", "role": "user"}]
    ]
    assert calls == [
        (
            "prompt_writing",
            response,
            sample_record()["fragments"],
            {"coder": ["f1"], "reviewer": ["f2"]},
            "f3",
        )
    ]
    assert eval_result.score == 0.0
    assert response_data[0]["task"] == PROMPT_TASK


def test_missing_scorer_dependency_fails_fast_before_sampling(tmp_path, monkeypatch):
    module = import_perspectivegap_module()
    monkeypatch.setattr(
        module,
        "_load_scoring_module",
        lambda: (_ for _ in ()).throw(ImportError(module.INSTALL_HINT)),
    )
    data_dir = write_dataset(tmp_path)
    sampler = RecordingSampler("this should not be requested")

    with pytest.raises(ImportError, match="PerspectiveGap scoring is required"):
        module.PerspectiveGapEval(
            task=ROLE_TASK,
            data_dir=str(data_dir),
            num_examples=1,
            proc_num=1,
        )(sampler)

    assert sampler.calls == []


def test_process_func_re_raises_missing_scorer_import_error(monkeypatch):
    module = import_perspectivegap_module()
    monkeypatch.setattr(
        module,
        "_load_scoring_module",
        lambda: (_ for _ in ()).throw(ImportError(module.INSTALL_HINT)),
    )
    sampler = RecordingSampler('{"coder": ["f1"], "reviewer": ["f2"]}')

    with pytest.raises(ImportError, match="PerspectiveGap scoring is required"):
        module.process_func(sampler, ROLE_TASK, sample_record())

    assert sampler.calls == []


def test_evaluate_registers_both_perspectivegap_tasks_in_cli_registry():
    evaluate_source = (Path(__file__).resolve().parent / "evaluate.py").read_text()

    assert "from evals.perspectivegap_eval import PerspectiveGapEval" in evaluate_source
    assert '"role_assignment": partial(' in evaluate_source
    assert 'task="role_assignment"' in evaluate_source
    assert '"prompt_writing": partial(' in evaluate_source
    assert 'task="prompt_writing"' in evaluate_source
