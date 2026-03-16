from __future__ import annotations

import logging
from typing import Any

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from pydantic import BaseModel, Field
from tenacity import (
    RetryError,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import get_settings
from app.logging_utils import setup_logging

setup_logging(level="INFO", module_name=__name__)
logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "qwen3.5-flash"
_DEFAULT_TIMEOUT_SECONDS = 60.0
_DEFAULT_MAX_RETRIES = 4


class LLMClientError(RuntimeError):
    """LLM 客户端调用异常。"""


class LLMRequestStats(BaseModel):
    """
    单次请求的基础统计信息。
    """

    message_count: int = Field(default=0, description="消息条数")
    has_system_message: bool = Field(default=False, description="是否包含 system message")
    temperature: float = Field(default=0.2, description="采样温度")


class LLMClient:
    """
    阿里云百炼 OpenAI 兼容接口客户端。

    职责边界：
    1. 仅负责模型调用；
    2. 不关心检索、提示词拼装、证据组织等上层逻辑；
    3. 统一处理超时、重试、错误日志和响应文本提取。
    """

    def __init__(
            self,
            *,
            base_url: str | None = None,
            api_key: str | None = None,
            model: str | None = None,
            timeout: float = _DEFAULT_TIMEOUT_SECONDS,
            max_retries: int = _DEFAULT_MAX_RETRIES,
    ) -> None:
        settings = get_settings()

        resolved_base_url = (base_url or settings.openai_base_url).strip()
        resolved_api_key = (api_key or settings.openai_api_key).strip()
        resolved_model = (model or settings.models.llm_model or _DEFAULT_MODEL).strip()

        if not resolved_base_url:
            raise LLMClientError("LLMClient 初始化失败：base_url 不能为空。")
        if not resolved_api_key:
            raise LLMClientError("LLMClient 初始化失败：api_key 不能为空。")
        if not resolved_model:
            raise LLMClientError("LLMClient 初始化失败：model 不能为空。")
        if timeout <= 0:
            raise LLMClientError("LLMClient 初始化失败：timeout 必须大于 0。")
        if max_retries <= 0:
            raise LLMClientError("LLMClient 初始化失败：max_retries 必须大于 0。")

        self.base_url = resolved_base_url
        self.api_key = resolved_api_key
        self.model = resolved_model
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)

        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

        logger.info(
            "LLMClient 初始化完成：base_url=%s, model=%s, timeout=%s, max_retries=%s",
            self.base_url,
            self.model,
            self.timeout,
            self.max_retries,
        )

    @staticmethod
    def _normalize_text(value: Any) -> str:
        """
        基础文本清洗：
        1. None 安全处理
        2. 全角空格转半角空格
        3. 去除首尾空白
        """
        if value is None:
            return ""
        return str(value).replace("\u3000", " ").strip()

    @staticmethod
    def _validate_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        校验并清洗消息列表。

        约束：
        - 必须为非空 list
        - 每个元素必须包含 role / content
        - role 必须是字符串
        - content 可为字符串或 SDK 可接受的结构；此处尽量不做破坏性转换
        """
        if not isinstance(messages, list):
            raise LLMClientError("messages 必须是 list[dict]。")
        if not messages:
            raise LLMClientError("messages 不能为空。")

        validated: list[dict[str, Any]] = []

        for idx, item in enumerate(messages):
            if not isinstance(item, dict):
                raise LLMClientError(f"messages 第 {idx} 项不是 dict。")

            role = item.get("role")
            if not isinstance(role, str) or not role.strip():
                raise LLMClientError(f"messages 第 {idx} 项缺少有效 role。")

            if "content" not in item:
                raise LLMClientError(f"messages 第 {idx} 项缺少 content。")

            content = item.get("content")
            if isinstance(content, str):
                cleaned_content = content.strip()
                if not cleaned_content:
                    raise LLMClientError(f"messages 第 {idx} 项 content 为空字符串。")
                validated.append({"role": role.strip(), "content": cleaned_content})
                continue

            # 对非字符串 content 保持透传，但仍要求非空
            if content is None:
                raise LLMClientError(f"messages 第 {idx} 项 content 不能为空。")

            cleaned_item = dict(item)
            cleaned_item["role"] = role.strip()
            validated.append(cleaned_item)

        return validated

    @staticmethod
    def _validate_temperature(temperature: float) -> float:
        """
        校验 temperature。
        """
        try:
            value = float(temperature)
        except (TypeError, ValueError) as exc:
            raise LLMClientError("temperature 必须是数值。") from exc

        if value < 0 or value > 2:
            raise LLMClientError("temperature 必须位于 [0, 2] 区间内。")

        return value

    @staticmethod
    def _build_request_stats(messages: list[dict[str, Any]], temperature: float) -> LLMRequestStats:
        """
        构造请求统计信息。
        """
        has_system = any(str(item.get("role", "")).strip().lower() == "system" for item in messages)
        return LLMRequestStats(
            message_count=len(messages),
            has_system_message=has_system,
            temperature=temperature,
        )

    @staticmethod
    def _extract_text_from_content(content: Any) -> str:
        """
        从 OpenAI 兼容响应的 message.content 中提取文本。

        兼容情况：
        1. content 为字符串
        2. content 为分段列表（如 text block）
        """
        if content is None:
            return ""

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    text = item.strip()
                    if text:
                        parts.append(text)
                    continue

                if isinstance(item, dict):
                    # 兼容常见格式：{"type": "text", "text": "..."}
                    text = item.get("text")
                    if isinstance(text, str) and text.strip():
                        parts.append(text.strip())
                        continue

                    # 某些兼容实现可能返回 {"type": "...", "content": "..."}
                    content_text = item.get("content")
                    if isinstance(content_text, str) and content_text.strip():
                        parts.append(content_text.strip())
                        continue

            return "\n".join(parts).strip()

        return str(content).strip()

    @staticmethod
    def _extract_answer_text(response: Any) -> str:
        """
        从 Chat Completions 响应对象中提取最终答案文本。
        """
        choices = getattr(response, "choices", None)
        if not choices:
            raise LLMClientError("模型响应异常：缺少 choices。")

        first_choice = choices[0]
        message = getattr(first_choice, "message", None)
        if message is None:
            raise LLMClientError("模型响应异常：缺少 message。")

        content = getattr(message, "content", None)
        text = LLMClient._extract_text_from_content(content)

        if text:
            return text

        # 某些兼容实现可能在 message 对象中返回其他文本字段
        fallback_fields = ["answer", "output_text"]
        for field_name in fallback_fields:
            value = getattr(message, field_name, None)
            fallback_text = LLMClient._extract_text_from_content(value)
            if fallback_text:
                return fallback_text

        raise LLMClientError("模型响应为空，未提取到有效文本。")

    @retry(
        retry=retry_if_exception_type(
            (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError, APIError)
        ),
        wait=wait_exponential(multiplier=1, min=1, max=16),
        stop=stop_after_attempt(_DEFAULT_MAX_RETRIES),
        reraise=True,
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _request_chat_completion(
            self,
            *,
            messages: list[dict[str, Any]],
            temperature: float,
    ) -> Any:
        """
        执行一次模型请求。

        说明：
        - enable_thinking 为百炼扩展参数，需要通过 extra_body 传入；
        - 当前客户端仅返回自然语言文本，不暴露底层响应对象给上层。
        """
        logger.debug(
            "开始调用 LLM：model=%s, message_count=%s, temperature=%s",
            self.model,
            len(messages),
            temperature,
        )

        return self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            extra_body={"enable_thinking": False},
        )

    def ask(self, messages: list[dict], temperature: float = 0.2) -> str:
        """
        调用模型生成答案，并返回纯文本结果。

        参数：
            messages:
                OpenAI Chat Completions 风格消息数组
            temperature:
                采样温度，默认 0.2，适合 FAQ / RAG 生成场景

        返回：
            str: 模型生成的答案文本
        """
        validated_messages = self._validate_messages(messages)
        validated_temperature = self._validate_temperature(temperature)
        stats = self._build_request_stats(validated_messages, validated_temperature)

        logger.info(
            "发送 LLM 请求：model=%s, message_count=%s, has_system=%s, temperature=%s",
            self.model,
            stats.message_count,
            stats.has_system_message,
            stats.temperature,
        )

        try:
            response = self._request_chat_completion(
                messages=validated_messages,
                temperature=validated_temperature,
            )
            answer = self._extract_answer_text(response)
        except RetryError as exc:
            logger.exception("LLM 请求在多次重试后仍失败。")
            raise LLMClientError("LLM 请求在多次重试后仍失败。") from exc
        except (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError, APIError) as exc:
            logger.exception("LLM 接口调用失败：%s", exc)
            raise LLMClientError("LLM 接口调用失败。") from exc
        except LLMClientError:
            logger.exception("LLM 响应解析失败。")
            raise
        except Exception as exc:
            logger.exception("LLM 调用出现未预期异常：%s", exc)
            raise LLMClientError("LLM 调用出现未预期异常。") from exc

        logger.info("LLM 请求完成：answer_length=%s", len(answer))
        return answer
