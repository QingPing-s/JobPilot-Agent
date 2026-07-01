from __future__ import annotations

import json
import os
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL_NAME = "deepseek-chat"
_ZERO_TOKEN_USAGE = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "calls": 0,
    "estimated_cost_usd": 0.0,
}
_TOKEN_USAGE: ContextVar[dict[str, int | float] | None] = ContextVar(
    "jobpilot_token_usage", default=None
)


def reset_token_usage() -> None:
    """Reset token usage counters for one JobPilot run."""
    _TOKEN_USAGE.set(_ZERO_TOKEN_USAGE.copy())


def get_token_usage() -> dict[str, int | float]:
    """Return accumulated token usage for the current JobPilot run."""
    usage = _TOKEN_USAGE.get()
    return usage.copy() if usage is not None else _ZERO_TOKEN_USAGE.copy()


def _usage_value(usage: Any, key: str) -> int:
    if usage is None:
        return 0
    if isinstance(usage, dict):
        value = usage.get(key, 0)
    else:
        value = getattr(usage, key, 0)
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _record_token_usage(response: Any) -> None:
    usage = getattr(response, "usage", None)
    current = get_token_usage()
    current["prompt_tokens"] += _usage_value(usage, "prompt_tokens")
    current["completion_tokens"] += _usage_value(usage, "completion_tokens")
    current["total_tokens"] += _usage_value(usage, "total_tokens")
    current["calls"] += 1
    try:
        input_rate = max(0.0, float(os.getenv("LLM_INPUT_COST_PER_MILLION", "0")))
        output_rate = max(0.0, float(os.getenv("LLM_OUTPUT_COST_PER_MILLION", "0")))
    except ValueError:
        input_rate, output_rate = 0.0, 0.0
    current["estimated_cost_usd"] = round(
        float(current.get("estimated_cost_usd", 0.0))
        + (_usage_value(usage, "prompt_tokens") * input_rate / 1_000_000)
        + (_usage_value(usage, "completion_tokens") * output_rate / 1_000_000),
        8,
    )
    _TOKEN_USAGE.set(current)

@dataclass
class LLMConfig:
    api_key: str | None = None
    base_url: str = DEFAULT_BASE_URL
    model_name: str = DEFAULT_MODEL_NAME
    timeout_seconds: float = 60.0
    max_retries: int = 1


class LLMClientError(RuntimeError):
    """Base error for LLM client failures."""


class MissingAPIKeyError(LLMClientError):
    """Raised when OPENAI_API_KEY is not configured."""


class LLMAPIError(LLMClientError):
    """Raised when the DeepSeek API call fails or returns an invalid response."""


class LLMJSONDecodeError(LLMClientError, ValueError):
    """Raised when a JSON response cannot be parsed into a dict."""


class DeepSeekClient:
    """Small DeepSeek wrapper using the OpenAI-compatible SDK."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout_seconds,
            max_retries=config.max_retries,
        )

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.2,
        response_format: dict[str, str] | None = None,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format

        try:
            response = self._client.chat.completions.create(**kwargs)
        except OpenAIError as exc:
            raise LLMAPIError(f"DeepSeek API 调用失败：{exc}") from exc
        except Exception as exc:
            raise LLMAPIError(f"LLM 客户端出现非预期错误：{exc}") from exc

        _record_token_usage(response)
        return _extract_text(response)


def get_llm_config() -> LLMConfig:
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is not None:
        api_key = api_key.strip()

    try:
        timeout_seconds = max(1.0, float(os.getenv("LLM_TIMEOUT_SECONDS", "60")))
    except ValueError:
        timeout_seconds = 60.0
    try:
        max_retries = max(0, min(3, int(os.getenv("LLM_SDK_MAX_RETRIES", "1"))))
    except ValueError:
        max_retries = 1

    return LLMConfig(
        api_key=api_key,
        base_url=os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL),
        model_name=os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME),
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
    )


def _validate_config(config: LLMConfig) -> None:
    if not config.api_key:
        raise MissingAPIKeyError("OPENAI_API_KEY 未配置，请在环境变量或 .env 文件中设置。")


def _extract_text(response: Any) -> str:
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError) as exc:
        raise LLMAPIError("DeepSeek API 返回结构不符合预期。") from exc

    if not isinstance(content, str) or not content.strip():
        raise LLMAPIError("DeepSeek API 返回了空内容或非文本内容。")

    return content.strip()


def build_client_from_env() -> DeepSeekClient:
    config = get_llm_config()
    _validate_config(config)
    return DeepSeekClient(config)


def call_llm(messages: list[dict], temperature: float = 0.2) -> str:
    """
    调用 DeepSeek Chat API，返回文本内容。
    """
    client = build_client_from_env()
    return client.chat(messages=messages, temperature=temperature)


def call_llm_json(messages: list[dict], temperature: float = 0.2) -> dict:
    """
    调用 DeepSeek Chat API，要求模型返回 JSON，并解析成 Python dict。
    如果解析失败，需要抛出清晰错误。
    """
    client = build_client_from_env()
    content = client.chat(
        messages=messages,
        temperature=temperature,
        response_format={"type": "json_object"},
    )

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        preview = content[:300].replace("\n", " ")
        raise LLMJSONDecodeError(f"LLM 返回内容无法解析为 JSON。响应预览：{preview}") from exc

    if not isinstance(parsed, dict):
        raise LLMJSONDecodeError("LLM 返回的 JSON 不是对象类型，无法转换为 dict。")

    return parsed
