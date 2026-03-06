"""Direct OpenAI-compatible provider — bypasses LiteLLM."""

from __future__ import annotations
from typing import Any

import httpx
import json_repair
from openai import AsyncOpenAI

from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from nanobot.providers.openai_codex_provider import (
    _consume_sse,
    _convert_messages,
    _convert_tools,
)


class CustomProvider(LLMProvider):

    def __init__(self, api_key: str = "no-key", api_base: str = "http://localhost:8000/v1", default_model: str = "default"):
        super().__init__(api_key, api_base)
        self.default_model = default_model
        self._client = AsyncOpenAI(api_key=api_key, base_url=api_base)

    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
                   model: str | None = None, max_tokens: int = 4096, temperature: float = 0.7,
                   reasoning_effort: str | None = None) -> LLMResponse:
        target_model = model or self.default_model
        clean_messages = self._sanitize_empty_content(messages)
        kwargs: dict[str, Any] = {
            "model": target_model,
            "messages": clean_messages,
            "max_tokens": max(1, max_tokens),
            "temperature": temperature,
        }
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        if tools:
            kwargs.update(tools=tools, tool_choice="auto")
        try:
            return self._parse(await self._client.chat.completions.create(**kwargs))
        except Exception as e:
            try:
                return await self._responses_chat(
                    messages=clean_messages,
                    tools=tools,
                    model=target_model,
                    max_tokens=max(1, max_tokens),
                    temperature=temperature,
                    reasoning_effort=reasoning_effort,
                )
            except Exception as fallback_error:
                return LLMResponse(
                    content=f"Error: {e}; responses fallback failed: {fallback_error}",
                    finish_reason="error",
                )

    def _parse(self, response: Any) -> LLMResponse:
        choice = response.choices[0]
        msg = choice.message
        tool_calls = [
            ToolCallRequest(id=tc.id, name=tc.function.name,
                            arguments=json_repair.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments)
            for tc in (msg.tool_calls or [])
        ]
        u = response.usage
        return LLMResponse(
            content=msg.content, tool_calls=tool_calls, finish_reason=choice.finish_reason or "stop",
            usage={"prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens, "total_tokens": u.total_tokens} if u else {},
            reasoning_content=getattr(msg, "reasoning_content", None) or None,
        )

    def get_default_model(self) -> str:
        return self.default_model

    async def _responses_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        model: str,
        max_tokens: int,
        temperature: float,
        reasoning_effort: str | None,
    ) -> LLMResponse:
        system_prompt, input_items = _convert_messages(messages)
        body: dict[str, Any] = {
            "model": model,
            "instructions": system_prompt or None,
            "input": input_items,
            "stream": True,
            "store": False,
            "max_output_tokens": max_tokens,
            "text": {"verbosity": "medium"},
            "tool_choice": "auto",
            "parallel_tool_calls": True,
            "temperature": temperature,
        }
        if reasoning_effort:
            body["reasoning"] = {"effort": reasoning_effort}
        if tools:
            body["tools"] = _convert_tools(tools)

        headers = {
            "Authorization": f"Bearer {self.api_key or 'no-key'}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        base_url = (self.api_base or "http://localhost:8000/v1").rstrip("/")
        url = f"{base_url}/responses"

        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", url, headers=headers, json=body) as response:
                if response.status_code != 200:
                    text = await response.aread()
                    raise RuntimeError(
                        f"responses endpoint returned {response.status_code}: "
                        f"{text.decode('utf-8', 'ignore')[:500]}"
                    )
                content, tool_calls, finish_reason = await _consume_sse(response)
                return LLMResponse(
                    content=content,
                    tool_calls=tool_calls,
                    finish_reason=finish_reason,
                )
