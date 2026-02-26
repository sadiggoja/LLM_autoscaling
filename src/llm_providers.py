import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import requests


@dataclass
class ToolCall:
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    tool_calls: list[ToolCall] = field(default_factory=list)
    text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0


class LLMProvider(ABC):
    @abstractmethod
    def query(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        pass


class AnthropicProvider(LLMProvider):
    """Claude API provider using the anthropic SDK with tool-use support."""

    # Cost per million tokens (Claude Sonnet 4)
    INPUT_COST_PER_M = 3.0
    OUTPUT_COST_PER_M = 15.0

    def __init__(self, model: str = "claude-sonnet-4-20250514", max_tokens: int = 1024,
                 api_key: Optional[str] = None):
        import anthropic
        self.model = model
        self.max_tokens = max_tokens
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    def query(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        start = time.time()
        try:
            kwargs = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "messages": messages,
            }
            if tools:
                kwargs["tools"] = tools

            response = self.client.messages.create(**kwargs)

            latency_ms = (time.time() - start) * 1000
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            cost = (input_tokens * self.INPUT_COST_PER_M + output_tokens * self.OUTPUT_COST_PER_M) / 1_000_000

            tool_calls = []
            text = ""
            for block in response.content:
                if block.type == "tool_use":
                    tool_calls.append(ToolCall(name=block.name, arguments=block.input))
                elif block.type == "text":
                    text = block.text

            return LLMResponse(
                tool_calls=tool_calls,
                text=text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                cost_usd=cost,
            )
        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            print(f"AnthropicProvider error: {e}")
            return LLMResponse(latency_ms=latency_ms)


class OllamaProvider(LLMProvider):
    """Local Llama provider via Ollama HTTP API."""

    def __init__(self, model: str = "llama3.1:8b", base_url: str = "http://localhost:11434",
                 max_tokens: int = 1024):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens

    def query(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        start = time.time()
        try:
            payload: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {"num_predict": self.max_tokens},
            }
            if tools:
                payload["tools"] = self._convert_tools(tools)

            resp = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()

            latency_ms = (time.time() - start) * 1000

            input_tokens = data.get("prompt_eval_count", 0)
            output_tokens = data.get("eval_count", 0)

            tool_calls = []
            text = ""

            message = data.get("message", {})
            if message.get("tool_calls"):
                for tc in message["tool_calls"]:
                    func = tc.get("function", {})
                    tool_calls.append(ToolCall(
                        name=func.get("name", ""),
                        arguments=func.get("arguments", {}),
                    ))
            if message.get("content"):
                text = message["content"]

            return LLMResponse(
                tool_calls=tool_calls,
                text=text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                cost_usd=0.0,  # Local inference has no API cost
            )
        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            print(f"OllamaProvider error: {e}")
            return LLMResponse(latency_ms=latency_ms)

    def _convert_tools(self, anthropic_tools: list[dict]) -> list[dict]:
        """Convert Anthropic-style tool definitions to Ollama format."""
        ollama_tools = []
        for tool in anthropic_tools:
            ollama_tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                },
            })
        return ollama_tools
