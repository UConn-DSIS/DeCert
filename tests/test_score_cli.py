import sys

from scoring.score_judgments import (
    _resolved_disable_thinking,
    _resolved_endpoint,
    _resolved_key_env,
    parse_args,
)


def test_scoring_cli_role_defaults(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "score_judgments.py",
            "--data",
            "data.jsonl",
            "--split-name",
            "train",
            "--rollouts",
            "rollouts.jsonl",
            "--out",
            "scored.jsonl",
        ],
    )
    args = parse_args()

    assert args.analyzer_model == "gpt-4o"
    assert args.verifier_model == "Qwen/Qwen3-14B"
    assert args.matcher_model == "Qwen/Qwen3-14B"


def test_qwen_endpoint_and_thinking_defaults():
    qwen = "Qwen/Qwen3-14B"
    qwen_url = _resolved_endpoint(qwen, None)
    assert qwen_url == "http://localhost:8000/v1"
    assert _resolved_key_env(qwen_url, None) == "VLLM_API_KEY"
    assert _resolved_disable_thinking(qwen, None) is True

    assert _resolved_endpoint("gpt-4o", None) is None
    assert _resolved_key_env(None, None) == "OPENAI_API_KEY"
    assert _resolved_disable_thinking("gpt-4o", None) is False
