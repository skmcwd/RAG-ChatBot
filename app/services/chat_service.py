from __future__ import annotations

import logging
import re
from typing import Any, Protocol

from app.clients.llm_client import LLMClient, LLMClientError
from app.models import ChatAnswer, EvidenceItem, RetrievedChunk
from app.retrieval.hybrid_retriever import HybridRetriever
from app.services.prompt_builder import build_messages

logger = logging.getLogger(__name__)

WHITESPACE_RE = re.compile(r"\s+")
DEFAULT_MAX_HISTORY_MESSAGES = 6
DEFAULT_MAX_EVIDENCE_ITEMS = 6
DEFAULT_GALLERY_IMAGE_LIMIT = 12
DEFAULT_SNIPPET_CHARS = 280
DEFAULT_RELIABLE_SCORE_THRESHOLD = 0.45
DEFAULT_MIN_USABLE_SCORE = 0.20


class ChatServiceError(RuntimeError):
    """ChatService 业务异常。"""


class PromptBuilderProtocol(Protocol):
    """
    PromptBuilder 适配协议。

    当前 prompt_builder.py 提供的是函数而非类。
    为保持后续可替换性，这里定义统一协议。
    """

    def build_messages(
            self,
            user_query: str,
            retrieved_chunks: list[RetrievedChunk],
    ) -> list[dict[str, str]]:
        """
        构造模型 messages。
        """
        ...


class _DefaultPromptBuilder:
    """
    默认 PromptBuilder 适配器。
    """

    def build_messages(
            self,
            user_query: str,
            retrieved_chunks: list[RetrievedChunk],
    ) -> list[dict[str, str]]:
        return build_messages(user_query=user_query, retrieved_chunks=retrieved_chunks)


class ChatService:
    """
    企业网银 FAQ RAG demo 的主业务服务层。

    职责：
    1. 调用 HybridRetriever 获取证据
    2. 调用 PromptBuilder 组织提示词
    3. 调用 LLMClient 生成答案
    4. 将检索结果整理为 EvidenceItem / gallery_images / debug_info
    5. 在证据不足时返回保守回复，避免无依据作答
    """

    def __init__(
            self,
            *,
            hybrid_retriever: HybridRetriever | None = None,
            prompt_builder: PromptBuilderProtocol | None = None,
            llm_client: LLMClient | None = None,
            reliable_score_threshold: float = DEFAULT_RELIABLE_SCORE_THRESHOLD,
            min_usable_score: float = DEFAULT_MIN_USABLE_SCORE,
            max_history_messages: int = DEFAULT_MAX_HISTORY_MESSAGES,
            max_evidence_items: int = DEFAULT_MAX_EVIDENCE_ITEMS,
            gallery_image_limit: int = DEFAULT_GALLERY_IMAGE_LIMIT,
    ) -> None:
        if reliable_score_threshold <= 0:
            raise ChatServiceError("reliable_score_threshold 必须大于 0。")
        if min_usable_score < 0:
            raise ChatServiceError("min_usable_score 不能小于 0。")
        if max_history_messages < 0:
            raise ChatServiceError("max_history_messages 不能小于 0。")
        if max_evidence_items <= 0:
            raise ChatServiceError("max_evidence_items 必须大于 0。")
        if gallery_image_limit <= 0:
            raise ChatServiceError("gallery_image_limit 必须大于 0。")

        self.hybrid_retriever = hybrid_retriever or HybridRetriever()
        self.prompt_builder = prompt_builder or _DefaultPromptBuilder()
        self.llm_client = llm_client or LLMClient()

        self.reliable_score_threshold = float(reliable_score_threshold)
        self.min_usable_score = float(min_usable_score)
        self.max_history_messages = int(max_history_messages)
        self.max_evidence_items = int(max_evidence_items)
        self.gallery_image_limit = int(gallery_image_limit)

    @staticmethod
    def _normalize_text(value: Any) -> str:
        """
        统一文本清洗：
        1. None 安全处理
        2. 全角空格转半角空格
        3. 连续空白折叠
        4. 去除首尾空白
        """
        if value is None:
            return ""

        text = str(value).replace("\u3000", " ")
        text = WHITESPACE_RE.sub(" ", text).strip()
        return text

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        """
        稳健转 float。
        """
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _unique_keep_order(items: list[str]) -> list[str]:
        """
        保持原顺序去重。
        """
        result: list[str] = []
        seen: set[str] = set()

        for item in items:
            text = ChatService._normalize_text(item)
            if not text:
                continue

            key = text.casefold()
            if key in seen:
                continue

            seen.add(key)
            result.append(text)

        return result

    @staticmethod
    def _truncate_text(text: str, max_chars: int = DEFAULT_SNIPPET_CHARS) -> str:
        """
        截断展示文本，避免前端证据区过长。
        """
        clean = ChatService._normalize_text(text)
        if len(clean) <= max_chars:
            return clean
        return clean[: max_chars - 1].rstrip() + "…"

    def _sanitize_history(self, history: list[dict] | None) -> list[dict[str, str]]:
        """
        清洗历史消息，仅保留 user / assistant 两类角色，便于直接送入模型。

        说明：
        - 为避免上下文膨胀，仅保留最近若干条
        - system 历史消息不透传，避免与当前系统提示冲突
        """
        if not history:
            return []

        if not isinstance(history, list):
            logger.warning("history 不是 list，已忽略。")
            return []

        sanitized: list[dict[str, str]] = []
        recent_history = history[-self.max_history_messages :] if self.max_history_messages > 0 else []

        for idx, item in enumerate(recent_history):
            if not isinstance(item, dict):
                logger.warning("history 第 %s 项不是 dict，已忽略。", idx)
                continue

            role = self._normalize_text(item.get("role")).lower()
            content = self._normalize_text(item.get("content"))

            if role not in {"user", "assistant"}:
                continue
            if not content:
                continue

            sanitized.append({"role": role, "content": content})

        return sanitized

    def _merge_history_into_messages(
            self,
            base_messages: list[dict[str, str]],
            history: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """
        将历史消息插入 prompt builder 生成的基础 messages 中。

        约定：
        - base_messages 通常为 [system, user]
        - 历史消息插入在 system 与当前 user 之间
        """
        if not base_messages:
            return history

        if not history:
            return base_messages

        if len(base_messages) == 1:
            return [base_messages[0], *history]

        return [base_messages[0], *history, *base_messages[1:]]

    def _is_reliable_evidence(self, retrieved_chunks: list[RetrievedChunk]) -> bool:
        """
        判断当前检索结果是否足以支持正常作答。

        判定原则：
        1. 至少存在一条可用证据
        2. 最高分达到阈值；或
        3. 存在明显精确命中 / 分类命中 / 整句命中等强信号
        """
        if not retrieved_chunks:
            return False

        top_chunk = retrieved_chunks[0]
        top_score = self._safe_float(top_chunk.retrieval_score, 0.0)
        if top_score >= self.reliable_score_threshold:
            return True

        if top_score < self.min_usable_score:
            return False

        reason = self._normalize_text(top_chunk.rerank_reason)
        strong_markers = ("精确命中", "整句命中", "分类命中")
        if any(marker in reason for marker in strong_markers):
            return True

        if len(retrieved_chunks) >= 2:
            second_score = self._safe_float(retrieved_chunks[1].retrieval_score, 0.0)
            if second_score >= self.min_usable_score:
                return True

        return False

    def _to_evidence_item(self, chunk: RetrievedChunk) -> EvidenceItem:
        """
        将 RetrievedChunk 映射为前端展示所需的 EvidenceItem。
        """
        snippet = self._truncate_text(chunk.full_text)
        return EvidenceItem(
            doc_id=self._normalize_text(chunk.doc_id),
            title=self._normalize_text(chunk.title),
            source_file=self._normalize_text(chunk.source_file),
            source_type=self._normalize_text(chunk.source_type),
            category=self._normalize_text(chunk.category) or None,
            snippet=snippet,
            quote=snippet if snippet else None,
            page_no=chunk.page_no,
            slide_no=chunk.slide_no,
            score=self._safe_float(chunk.retrieval_score, 0.0),
            reason=self._normalize_text(chunk.rerank_reason) or None,
            image_paths=self._unique_keep_order(list(chunk.image_paths or [])),
        )

    def _build_evidence_items(self, retrieved_chunks: list[RetrievedChunk]) -> list[EvidenceItem]:
        """
        构造 EvidenceItem 列表。
        """
        items = [self._to_evidence_item(chunk) for chunk in retrieved_chunks[: self.max_evidence_items]]
        return items

    def _collect_gallery_images(self, retrieved_chunks: list[RetrievedChunk]) -> list[str]:
        """
        聚合去重后的图片列表，供 Gradio Gallery 直接使用。
        """
        images: list[str] = []
        for chunk in retrieved_chunks:
            images.extend(list(chunk.image_paths or []))

        images = self._unique_keep_order(images)
        return images[: self.gallery_image_limit]

    def _build_conservative_answer(self, query: str, has_evidence: bool) -> str:
        """
        在证据不足或模型调用失败时生成保守回复。
        """
        query_text = self._normalize_text(query)

        if has_evidence:
            return f"""一、结论
根据当前知识库无法确认该问题的完整处理结论。

二、操作步骤
1. 请优先查看下方证据区中的相关资料原文。
2. 若你能补充更具体的信息，例如报错原文、错误码、菜单路径、页面截图、操作角色或当前步骤，我可以基于更明确的线索继续检索。
3. 当前问题：{query_text}

三、补充说明
1. 是否需要柜面处理，根据当前知识库无法确认。
2. 当前已检索到部分相关资料，但不足以支持稳定、明确的最终答复。"""

        return f"""一、结论
根据当前知识库无法确认。

二、操作步骤
1. 请补充更具体的信息，例如报错原文、错误码、页面截图、菜单路径、功能名称、操作角色或当前执行到哪一步。
2. 若问题与证书、UKey、权限、回单、转账或控件有关，请尽量提供界面提示原文。
3. 当前问题：{query_text}

三、补充说明
1. 当前知识库中未检索到足够可靠的相关证据。
2. 是否需要柜面处理，根据当前知识库无法确认。"""

    def _build_fallback_answer_from_evidence(
            self,
            query: str,
            evidence_items: list[EvidenceItem],
    ) -> str:
        """
        当 LLM 调用失败但已有证据时，返回一个不夸张的降级答复。
        """
        if not evidence_items:
            return self._build_conservative_answer(query=query, has_evidence=False)

        references = []
        for idx, item in enumerate(evidence_items[:3], start=1):
            title = self._normalize_text(item.title) or "未命名资料"
            source_file = self._normalize_text(item.source_file) or "未知文件"
            references.append(f"{idx}. {title}（{source_file}）")

        reference_text = "\n".join(references)

        return f"""一、结论
已检索到与问题相关的知识库资料，但自动生成答案失败；根据当前知识库无法确认更完整结论。

二、操作步骤
1. 请优先查看下方证据区中的原文内容。
2. 可重点参考以下资料：
{reference_text}

三、补充说明
1. 是否需要柜面处理，根据当前知识库无法确认。
2. 当前返回的是降级答复，建议结合证据原文进一步核对。"""

    def _build_debug_info(
            self,
            query: str,
            retrieved_chunks: list[RetrievedChunk],
            retriever_debug: dict[str, Any],
            *,
            used_conservative_answer: bool,
            llm_called: bool,
            llm_succeeded: bool,
            error_message: str | None = None,
    ) -> dict[str, Any]:
        """
        组织 debug_info，便于调试与 UI 展示。
        """
        normalized_info = retriever_debug.get("normalized", {}) if isinstance(retriever_debug, dict) else {}
        normalized_query = self._normalize_text(normalized_info.get("normalized_query", ""))

        top_results = []
        scoring_reasons = []

        for chunk in retrieved_chunks[: self.max_evidence_items]:
            score = self._safe_float(chunk.retrieval_score, 0.0)
            reason = self._normalize_text(chunk.rerank_reason)
            row = {
                "doc_id": self._normalize_text(chunk.doc_id),
                "title": self._normalize_text(chunk.title),
                "source_type": self._normalize_text(chunk.source_type),
                "source_file": self._normalize_text(chunk.source_file),
                "category": self._normalize_text(chunk.category) or None,
                "score": score,
                "reason": reason,
                "has_images": bool(chunk.image_paths),
            }
            top_results.append(row)
            scoring_reasons.append(
                {
                    "doc_id": row["doc_id"],
                    "title": row["title"],
                    "score": score,
                    "reason": reason,
                }
            )

        debug_info: dict[str, Any] = {
            "query": self._normalize_text(query),
            "normalized_query": normalized_query,
            "top_results": top_results,
            "scoring_reasons": scoring_reasons,
            "retriever_debug": retriever_debug if isinstance(retriever_debug, dict) else {},
            "used_conservative_answer": used_conservative_answer,
            "llm_called": llm_called,
            "llm_succeeded": llm_succeeded,
        }

        if error_message:
            debug_info["error"] = error_message

        return debug_info

    def chat(self, query: str, history: list[dict] | None = None) -> ChatAnswer:
        """
        执行一次完整 RAG 对话流程，并返回 ChatAnswer。

        流程：
        1. 检索 top-k 证据
        2. 判断证据是否可靠
        3. 生成 prompt messages
        4. 调用 LLM 生成答案
        5. 整理 EvidenceItem / gallery_images / debug_info
        """
        clean_query = self._normalize_text(query)
        if not clean_query:
            conservative = self._build_conservative_answer(query="", has_evidence=False)
            return ChatAnswer(
                answer_markdown=conservative,
                evidence_items=[],
                gallery_images=[],
                debug_info={
                    "query": "",
                    "normalized_query": "",
                    "top_results": [],
                    "scoring_reasons": [],
                    "used_conservative_answer": True,
                    "llm_called": False,
                    "llm_succeeded": False,
                    "error": "empty_query",
                },
            )

        sanitized_history = self._sanitize_history(history)
        retrieved_chunks: list[RetrievedChunk] = []
        retriever_debug: dict[str, Any] = {}
        evidence_items: list[EvidenceItem] = []
        gallery_images: list[str] = []
        llm_called = False
        llm_succeeded = False

        try:
            retrieved_chunks = self.hybrid_retriever.retrieve(clean_query)
            retriever_debug = self.hybrid_retriever.get_last_debug_info()
        except Exception as exc:
            logger.exception("HybridRetriever 执行失败：%s", exc)
            answer = self._build_conservative_answer(query=clean_query, has_evidence=False)
            return ChatAnswer(
                answer_markdown=answer,
                evidence_items=[],
                gallery_images=[],
                debug_info={
                    "query": clean_query,
                    "normalized_query": "",
                    "top_results": [],
                    "scoring_reasons": [],
                    "used_conservative_answer": True,
                    "llm_called": False,
                    "llm_succeeded": False,
                    "error": f"retriever_error:{exc}",
                },
            )

        evidence_items = self._build_evidence_items(retrieved_chunks)
        gallery_images = self._collect_gallery_images(retrieved_chunks)
        reliable = self._is_reliable_evidence(retrieved_chunks)

        if not reliable:
            logger.info("未检索到可靠证据，返回保守答复：query=%s", clean_query)
            answer = self._build_conservative_answer(
                query=clean_query,
                has_evidence=bool(retrieved_chunks),
            )
            debug_info = self._build_debug_info(
                query=clean_query,
                retrieved_chunks=retrieved_chunks,
                retriever_debug=retriever_debug,
                used_conservative_answer=True,
                llm_called=False,
                llm_succeeded=False,
            )
            return ChatAnswer(
                answer_markdown=answer,
                evidence_items=evidence_items,
                gallery_images=gallery_images,
                debug_info=debug_info,
            )

        try:
            base_messages = self.prompt_builder.build_messages(
                user_query=clean_query,
                retrieved_chunks=retrieved_chunks,
            )
            messages = self._merge_history_into_messages(
                base_messages=base_messages,
                history=sanitized_history,
            )
        except Exception as exc:
            logger.exception("PromptBuilder 执行失败：%s", exc)
            answer = self._build_fallback_answer_from_evidence(
                query=clean_query,
                evidence_items=evidence_items,
            )
            debug_info = self._build_debug_info(
                query=clean_query,
                retrieved_chunks=retrieved_chunks,
                retriever_debug=retriever_debug,
                used_conservative_answer=True,
                llm_called=False,
                llm_succeeded=False,
                error_message=f"prompt_builder_error:{exc}",
            )
            return ChatAnswer(
                answer_markdown=answer,
                evidence_items=evidence_items,
                gallery_images=gallery_images,
                debug_info=debug_info,
            )

        try:
            llm_called = True
            answer_text = self.llm_client.ask(messages=messages, temperature=0.2)
            llm_succeeded = True

            debug_info = self._build_debug_info(
                query=clean_query,
                retrieved_chunks=retrieved_chunks,
                retriever_debug=retriever_debug,
                used_conservative_answer=False,
                llm_called=True,
                llm_succeeded=True,
            )
            debug_info["history_message_count"] = len(sanitized_history)
            debug_info["llm_message_count"] = len(messages)
            debug_info["llm_answer_length"] = len(answer_text)

            return ChatAnswer(
                answer_markdown=answer_text,
                evidence_items=evidence_items,
                gallery_images=gallery_images,
                debug_info=debug_info,
            )

        except LLMClientError as exc:
            logger.exception("LLM 调用失败：%s", exc)
            answer = self._build_fallback_answer_from_evidence(
                query=clean_query,
                evidence_items=evidence_items,
            )
            debug_info = self._build_debug_info(
                query=clean_query,
                retrieved_chunks=retrieved_chunks,
                retriever_debug=retriever_debug,
                used_conservative_answer=True,
                llm_called=llm_called,
                llm_succeeded=False,
                error_message=f"llm_error:{exc}",
            )
            debug_info["history_message_count"] = len(sanitized_history)
            debug_info["llm_message_count"] = len(messages)

            return ChatAnswer(
                answer_markdown=answer,
                evidence_items=evidence_items,
                gallery_images=gallery_images,
                debug_info=debug_info,
            )

        except Exception as exc:
            logger.exception("ChatService 出现未预期异常：%s", exc)
            answer = self._build_fallback_answer_from_evidence(
                query=clean_query,
                evidence_items=evidence_items,
            )
            debug_info = self._build_debug_info(
                query=clean_query,
                retrieved_chunks=retrieved_chunks,
                retriever_debug=retriever_debug,
                used_conservative_answer=True,
                llm_called=llm_called,
                llm_succeeded=False,
                error_message=f"unexpected_error:{exc}",
            )
            debug_info["history_message_count"] = len(sanitized_history)

            return ChatAnswer(
                answer_markdown=answer,
                evidence_items=evidence_items,
                gallery_images=gallery_images,
                debug_info=debug_info,
            )