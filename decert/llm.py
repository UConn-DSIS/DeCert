"""Minimal LLM client abstraction.

Every component that calls a model takes an `LLMClient`, so judges, verifiers
and analyzers can be swapped between OpenAI, a local vLLM server (OpenAI
compatible), or a FakeClient in tests.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from .types import Message


class LLMClient(Protocol):
    def complete(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 2048,
        seed: int | None = None,
    ) -> str: ...


@dataclass
class OpenAICompatClient:
    """Chat-completions client for OpenAI or any OpenAI-compatible server
    (e.g. `vllm serve Qwen/Qwen3-8B` with --port and base_url
    http://localhost:8000/v1)."""

    model: str
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    # For reasoning models served by vLLM (e.g. Qwen3): turn off thinking so
    # the token budget goes to the certificate, not a <think> block.
    disable_thinking: bool = False
    # For gpt-oss models: cap the reasoning effort ("low"/"medium"/"high").
    # Reasoning tokens count toward max_tokens, so long verifier prompts with
    # medium+ effort can exhaust the budget before the final answer starts.
    reasoning_effort: str | None = None
    _client: object = field(default=None, repr=False)

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            api_key = os.environ.get(self.api_key_env) or "EMPTY"
            self._client = OpenAI(base_url=self.base_url, api_key=api_key)
        return self._client

    def complete(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 2048,
        seed: int | None = None,
    ) -> str:
        response = self._get_client().chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed,
            **self._extra(),
        )
        return response.choices[0].message.content or ""

    def _extra(self) -> dict:
        extra_body = {}
        if self.disable_thinking:
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}
        if self.reasoning_effort:
            extra_body["reasoning_effort"] = self.reasoning_effort
        return {"extra_body": extra_body} if extra_body else {}


@dataclass
class FakeClient:
    """Test double: pops queued responses, or delegates to a handler fn."""

    responses: list[str] = field(default_factory=list)
    handler: Callable[[list[Message]], str] | None = None
    calls: list = field(default_factory=list)

    def complete(self, messages: list[Message], **_kwargs) -> str:
        self.calls.append(messages)
        if self.handler is not None:
            return self.handler(messages)
        return self.responses.pop(0)
