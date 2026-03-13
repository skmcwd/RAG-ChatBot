from __future__ import annotations

import logging
import math
import re
from typing import Sequence

from openai import APIConnectionError, APIError, APITimeoutError, InternalServerError, OpenAI, RateLimitError
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

logger = logging.getLogger(__name__)

_WHITESPACE_RE = re.compile(r"\s+")
_DEFAULT_MAX_TEXT_CHARS = 6000
_DEFAULT_BATCH_SIZE = 64
_DEFAULT_TIMEOUT_SECONDS = 60.0


class EmbeddingClientError(RuntimeError):
    """Embedding 客户端异常。"""


class EmbeddingRequestStats(BaseModel):
    """
    单次请求的简单统计信息。
    """

    requested_count: int = Field(default=0, description="请求的文本条数")
    empty_count: int = Field(default=0, description="空文本条数")
    truncated_count: int = Field(default=0, description="被截断文本条数")


class EmbeddingClient:
    """
    阿里云百炼 OpenAI 兼容 Embedding 客户端。

    设计目标：
    1. 默认从项目配置读取 base_url / api_key / model
    2. 支持单条与批量向量化
    3. 对空文本、超长文本、网络抖动和服务端瞬时异常做稳健处理
    4. 返回结果结构清晰，便于后续写入 Chroma 等向量库
    """

    def __init__(
            self,
            *,
            base_url: str | None = None,
            api_key: str | None = None,
            model: str | None = None,
            max_text_chars: int = _DEFAULT_MAX_TEXT_CHARS,
            batch_size: int = _DEFAULT_BATCH_SIZE,
            timeout: float = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        settings = get_settings()

        resolved_base_url = (base_url or settings.openai_base_url).strip()
        resolved_api_key = (api_key or settings.openai_api_key).strip()
        resolved_model = (model or settings.models.embed_model or "text-embedding-v4").strip()

        if not resolved_base_url:
            raise EmbeddingClientError("Embedding 客户端初始化失败：base_url 不能为空。")
        if not resolved_api_key:
            raise EmbeddingClientError("Embedding 客户端初始化失败：api_key 不能为空。")
        if not resolved_model:
            raise EmbeddingClientError("Embedding 客户端初始化失败：model 不能为空。")
        if max_text_chars <= 0:
            raise EmbeddingClientError("Embedding 客户端初始化失败：max_text_chars 必须大于 0。")
        if batch_size <= 0:
            raise EmbeddingClientError("Embedding 客户端初始化失败：batch_size 必须大于 0。")
        if timeout <= 0:
            raise EmbeddingClientError("Embedding 客户端初始化失败：timeout 必须大于 0。")

        self.base_url = resolved_base_url
        self.api_key = resolved_api_key
        self.model = resolved_model
        self.max_text_chars = int(max_text_chars)
        self.batch_size = int(batch_size)
        self.timeout = float(timeout)

        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

        logger.info(
            "EmbeddingClient 初始化完成：base_url=%s, model=%s, batch_size=%s, max_text_chars=%s",
            self.base_url,
            self.model,
            self.batch_size,
            self.max_text_chars,
        )

    @staticmethod
    def _normalize_text(text: str | None) -> str:
        """
        统一清洗文本：
        1. None 安全处理
        2. 全角空格转半角空格
        3. 连续空白压缩为单空格
        4. 去除首尾空白
        """
        if text is None:
            return ""
        normalized = str(text).replace("\u3000", " ")
        normalized = _WHITESPACE_RE.sub(" ", normalized).strip()
        return normalized

    def _truncate_text(self, text: str) -> tuple[str, bool]:
        """
        对超长文本做保守截断，避免请求过大。
        返回：
            (处理后的文本, 是否发生截断)
        """
        if len(text) <= self.max_text_chars:
            return text, False
        truncated = text[: self.max_text_chars].rstrip()
        return truncated, True

    def _prepare_single_text(self, text: str | None) -> tuple[str, bool, bool]:
        """
        准备单条文本。
        返回：
            (清洗后文本, 是否为空文本, 是否发生截断)
        """
        normalized = self._normalize_text(text)
        if not normalized:
            return "", True, False

        truncated, was_truncated = self._truncate_text(normalized)
        return truncated, False, was_truncated

    def _prepare_batch_texts(
            self,
            texts: Sequence[str],
    ) -> tuple[list[str], list[int], EmbeddingRequestStats]:
        """
        批量准备输入文本。

        返回：
            prepared_texts: 实际送入接口的文本列表
            valid_indices: prepared_texts 对应原始 texts 的下标
            stats: 预处理统计
        """
        prepared_texts: list[str] = []
        valid_indices: list[int] = []
        stats = EmbeddingRequestStats()

        for idx, text in enumerate(texts):
            cleaned, is_empty, was_truncated = self._prepare_single_text(text)
            stats.requested_count += 1

            if is_empty:
                stats.empty_count += 1
                continue

            if was_truncated:
                stats.truncated_count += 1
                logger.warning(
                    "检测到超长文本，已执行截断：index=%s, original_len=%s, truncated_len=%s",
                    idx,
                    len(self._normalize_text(text)),
                    len(cleaned),
                )

            prepared_texts.append(cleaned)
            valid_indices.append(idx)

        return prepared_texts, valid_indices, stats

    @retry(
        retry=retry_if_exception_type(
            (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError, APIError)
        ),
        wait=wait_exponential(multiplier=1, min=1, max=16),
        stop=stop_after_attempt(4),
        reraise=True,
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _request_embeddings(self, inputs: list[str]) -> list[list[float]]:
        """
        调用 OpenAI 兼容 Embedding 接口。

        仅处理“已校验且非空”的文本。
        """
        if not inputs:
            return []

        logger.debug("开始请求 Embedding：count=%s, model=%s", len(inputs), self.model)

        response = self._client.embeddings.create(
            model=self.model,
            input=inputs,
            encoding_format="float",
        )

        if not hasattr(response, "data") or response.data is None:
            raise EmbeddingClientError("Embedding 接口返回异常：缺少 data 字段。")

        items = sorted(response.data, key=lambda item: getattr(item, "index", 0))
        vectors: list[list[float]] = []

        for i, item in enumerate(items):
            embedding = getattr(item, "embedding", None)
            if not isinstance(embedding, list) or not embedding:
                raise EmbeddingClientError(
                    f"Embedding 接口返回异常：第 {i} 条结果缺少有效 embedding。"
                )

            vector = [float(x) for x in embedding]
            vectors.append(vector)

        if len(vectors) != len(inputs):
            raise EmbeddingClientError(
                f"Embedding 返回条数不匹配：请求 {len(inputs)} 条，返回 {len(vectors)} 条。"
            )

        logger.debug("Embedding 请求完成：count=%s, dim=%s", len(vectors), len(vectors[0]) if vectors else 0)
        return vectors

    def embed_text(self, text: str) -> list[float]:
        """
        生成单条文本的向量。

        处理规则：
        - 空文本：返回空列表，并记录警告日志
        - 超长文本：自动截断
        - API 异常：进行指数退避重试，最终失败时抛出 EmbeddingClientError
        """
        cleaned, is_empty, was_truncated = self._prepare_single_text(text)

        if is_empty:
            logger.warning("收到空文本，返回空向量。")
            return []

        if was_truncated:
            logger.warning("单条文本超过长度限制，已自动截断。")

        try:
            vectors = self._request_embeddings([cleaned])
        except RetryError as exc:
            raise EmbeddingClientError("Embedding 接口多次重试后仍失败。") from exc
        except Exception as exc:
            raise EmbeddingClientError("Embedding 接口调用失败。") from exc

        if not vectors or not vectors[0]:
            logger.warning("Embedding 接口返回空结果。")
            return []

        return vectors[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        批量生成文本向量，并保持与输入顺序一致。

        处理规则：
        - 对空文本位置返回空列表 []
        - 对超长文本自动截断
        - 按 batch_size 分批请求
        - 请求异常时进行指数退避重试，最终失败时抛出 EmbeddingClientError

        返回：
            与输入 texts 等长的二维列表。
            对于空文本，其对应位置为 []。
        """
        if not isinstance(texts, list):
            raise EmbeddingClientError("embed_texts 输入必须是 list[str]。")

        if not texts:
            logger.info("embed_texts 收到空列表，直接返回空结果。")
            return []

        prepared_texts, valid_indices, stats = self._prepare_batch_texts(texts)
        results: list[list[float]] = [[] for _ in texts]

        logger.info(
            "开始批量生成 Embedding：total=%s, valid=%s, empty=%s, truncated=%s, batch_size=%s",
            stats.requested_count,
            len(prepared_texts),
            stats.empty_count,
            stats.truncated_count,
            self.batch_size,
        )

        if not prepared_texts:
            logger.warning("批量输入全部为空文本，返回等长空向量列表。")
            return results

        all_vectors: list[list[float]] = []
        total_batches = math.ceil(len(prepared_texts) / self.batch_size)

        for batch_idx in range(total_batches):
            start = batch_idx * self.batch_size
            end = min(start + self.batch_size, len(prepared_texts))
            batch_inputs = prepared_texts[start:end]

            logger.debug(
                "发送 Embedding 批请求：batch=%s/%s, size=%s",
                batch_idx + 1,
                total_batches,
                len(batch_inputs),
                )

            try:
                batch_vectors = self._request_embeddings(batch_inputs)
            except RetryError as exc:
                raise EmbeddingClientError(
                    f"Embedding 接口批量请求失败：batch={batch_idx + 1}/{total_batches}。"
                ) from exc
            except Exception as exc:
                raise EmbeddingClientError(
                    f"Embedding 接口批量请求异常：batch={batch_idx + 1}/{total_batches}。"
                ) from exc

            if len(batch_vectors) != len(batch_inputs):
                raise EmbeddingClientError(
                    f"批量 Embedding 返回条数不匹配：batch={batch_idx + 1}/{total_batches}，"
                    f"请求 {len(batch_inputs)} 条，返回 {len(batch_vectors)} 条。"
                )

            all_vectors.extend(batch_vectors)

        if len(all_vectors) != len(valid_indices):
            raise EmbeddingClientError(
                f"批量 Embedding 结果总数不匹配：有效输入 {len(valid_indices)} 条，"
                f"实际返回 {len(all_vectors)} 条。"
            )

        for src_idx, vector in zip(valid_indices, all_vectors, strict=True):
            results[src_idx] = vector

        logger.info(
            "批量 Embedding 完成：total=%s, success=%s, empty=%s",
            len(texts),
            len(valid_indices),
            stats.empty_count,
        )
        return results