from __future__ import annotations

import logging
import re
from typing import Any

from app.models import RetrievedChunk

logger = logging.getLogger(__name__)

WHITESPACE_RE = re.compile(r"\s+")
DEFAULT_MAX_CHUNKS = 6
DEFAULT_MAX_CHUNK_TEXT_CHARS = 1800
DEFAULT_MAX_TOTAL_CONTEXT_CHARS = 9000

SYSTEM_PROMPT = """你是企业网银 FAQ 助手。

回答约束：
1. 只能依据给定资料回答，不得使用资料之外的假设、常识补全或自行推断业务规则。
2. 若资料不足、资料冲突或无法明确支撑结论，必须明确写出：“根据当前知识库无法确认”。
3. 优先回答以下内容：
   - 操作路径
   - 排查步骤
   - 是否需要柜面处理
4. 不得编造未出现的菜单、错误码、流程、页面名称、按钮名称、角色权限或处理时效。
5. 若资料中存在多个可能场景，应清晰区分适用条件，不得混为一谈。
6. 回答应简洁、准确、面向实际操作。

输出格式必须严格使用以下三级标题，且保持顺序不变：
一、结论
二、操作步骤
三、补充说明
"""


def _normalize_text(value: Any) -> str:
    """
    统一清洗文本：
    1. None 安全处理
    2. 全角空格转半角空格
    3. 连续空白折叠为单空格
    4. 去除首尾空白
    """
    if value is None:
        return ""

    text = str(value).replace("\u3000", " ")
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def _safe_int(value: Any) -> int | None:
    """
    稳健转换为 int。
    """
    if value is None:
        return None

    text = _normalize_text(value)
    if not text:
        return None

    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def _truncate_text(text: str, max_chars: int) -> str:
    """
    对长文本做温和截断，避免 prompt 过长。
    """
    clean = _normalize_text(text)
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 1].rstrip() + "…"


def _format_location(chunk: RetrievedChunk) -> str:
    """
    格式化页码/页号信息。
    """
    if chunk.slide_no is not None:
        slide_no = _safe_int(chunk.slide_no)
        if slide_no is not None:
            return f"slide_no={slide_no}"

    if chunk.page_no is not None:
        page_no = _safe_int(chunk.page_no)
        if page_no is not None:
            return f"page_no={page_no}"

    return "slide_no/page_no=未知"


def _format_source_line(chunk: RetrievedChunk) -> str:
    """
    格式化证据来源说明。
    """
    source_file = _normalize_text(chunk.source_file) or "未知文件"
    source_type = _normalize_text(chunk.source_type) or "未知类型"
    title = _normalize_text(chunk.title) or "未命名片段"
    category = _normalize_text(chunk.category) or "未分类"
    location = _format_location(chunk)

    return (
        f"source_file={source_file} | "
        f"source_type={source_type} | "
        f"title={title} | "
        f"category={category} | "
        f"{location}"
    )


def _format_single_evidence(
        chunk: RetrievedChunk,
        *,
        index: int,
        max_text_chars: int,
) -> str:
    """
    将单条 RetrievedChunk 格式化为结构化证据块文本。
    """
    body = _truncate_text(chunk.full_text, max_text_chars)
    retrieval_score = float(chunk.retrieval_score)
    rerank_reason = _normalize_text(chunk.rerank_reason) or "无"

    lines = [
        f"[证据 {index}]",
        _format_source_line(chunk),
        f"retrieval_score={retrieval_score:.6f}",
        f"rerank_reason={rerank_reason}",
        "正文：",
        body or "（无正文）",
    ]

    return "\n".join(lines)


def _build_context_block(
        retrieved_chunks: list[RetrievedChunk],
        *,
        max_chunks: int,
        max_chunk_text_chars: int,
        max_total_context_chars: int,
) -> str:
    """
    将 top-k 检索结果拼装成用户上下文中的证据区块。

    处理原则：
    1. 保持结果顺序
    2. 单条证据限制长度
    3. 总上下文长度限制，避免 prompt 失控
    """
    if not retrieved_chunks:
        return "未检索到可用证据。"

    evidence_blocks: list[str] = []
    total_chars = 0

    for idx, chunk in enumerate(retrieved_chunks[:max_chunks], start=1):
        block = _format_single_evidence(
            chunk,
            index=idx,
            max_text_chars=max_chunk_text_chars,
        )

        projected = total_chars + len(block)
        if evidence_blocks and projected > max_total_context_chars:
            logger.debug(
                "Prompt 证据区达到长度上限，停止追加：current=%s, next=%s, limit=%s",
                total_chars,
                len(block),
                max_total_context_chars,
            )
            break

        evidence_blocks.append(block)
        total_chars += len(block)

    if not evidence_blocks:
        return "未检索到可用证据。"

    return "\n\n".join(evidence_blocks)


def build_messages(
        user_query: str,
        retrieved_chunks: list[RetrievedChunk],
) -> list[dict[str, str]]:
    """
    构造发送给 LLM 的 messages。

    参数：
        user_query:
            用户原始问题
        retrieved_chunks:
            已排序的检索结果列表，建议外部已按 top-k 截断

    返回：
        list[dict[str, str]]:
            OpenAI Chat Completions 兼容的消息结构
    """
    query = _normalize_text(user_query)
    if not query:
        raise ValueError("user_query 不能为空。")

    context_block = _build_context_block(
        retrieved_chunks=retrieved_chunks,
        max_chunks=DEFAULT_MAX_CHUNKS,
        max_chunk_text_chars=DEFAULT_MAX_CHUNK_TEXT_CHARS,
        max_total_context_chars=DEFAULT_MAX_TOTAL_CONTEXT_CHARS,
    )

    user_prompt = f"""以下是用户问题与可用资料，请你严格基于资料回答。

【用户问题】
{query}

【可用资料】
{context_block}

请严格按照以下格式输出，不得增删标题：

一、结论
- 先直接回答用户问题。
- 若资料不足或无法确认，请明确写出：根据当前知识库无法确认。

二、操作步骤
- 优先给出菜单路径、点击位置、排查顺序。
- 若资料中没有明确步骤，请写“根据当前知识库无法确认具体操作步骤”。

三、补充说明
- 说明适用条件、常见限制、是否需要柜面处理。
- 若资料未提及柜面处理，不要自行判断需要或不需要柜面处理。
"""

    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    logger.debug(
        "Prompt 构造完成：query_len=%s, evidence_count=%s, user_prompt_len=%s",
        len(query),
        len(retrieved_chunks),
        len(user_prompt),
    )
    return messages
