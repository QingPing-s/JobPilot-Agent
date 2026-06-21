from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError


DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL_NAME = "deepseek-chat"

@dataclass
class LLMConfig:
    api_key: str | None = None
    base_url: str = DEFAULT_BASE_URL
    model_name: str = DEFAULT_MODEL_NAME


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
        self._client = OpenAI(api_key=config.api_key, base_url=config.base_url)

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

        return _extract_text(response)


def get_llm_config() -> LLMConfig:
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is not None:
        api_key = api_key.strip()

    return LLMConfig(
        api_key=api_key,
        base_url=os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL),
        model_name=os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME),
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
