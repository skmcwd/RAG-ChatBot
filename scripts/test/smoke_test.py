from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

# 兼容直接执行：
# python scripts/smoke_test.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.logging_utils import setup_logging  # noqa: E402
from app.models import ChatAnswer, EvidenceItem  # noqa: E402
from app.services.chat_service import ChatService, ChatServiceError  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_TEST_QUERIES: list[str] = [
    "代发工资失败后在哪里查询失败原因？",
    "UKey 插入电脑后企业网银无法识别怎么办？",
    "提示用户暂无权限，应该如何处理？",
    "回单列表里出现红叉是什么意思，怎么处理？",
    "企业网银控件为什么总是重复下载？",
    "页面报错 undefined message 怎么办？",
    "证书下载或证书初始化失败应该怎么排查？",
]

SEPARATOR = "=" * 88
SUB_SEPARATOR = "-" * 88


def _normalize_text(value: Any) -> str:
    """
    统一文本清洗：
    1. None 安全处理
    2. 全角空格转半角空格
    3. 去除首尾空白
    """
    if value is None:
        return ""
    return str(value).replace("\u3000", " ").strip()


def _safe_float(value: Any, default: float = 0.0) -> float:
    """
    稳健转换为 float。
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _build_answer_summary(answer_markdown: str, max_chars: int = 220) -> str:
    """
    从最终回答中提取简短摘要，便于在命令行快速浏览。

    处理原则：
    1. 优先提取“结论”部分
    2. 若无法识别结构，则退化为取前若干字符
    """
    text = _normalize_text(answer_markdown)
    if not text:
        return "（空答案）"

    markers = ["一、结论", "二、操作步骤", "三、补充说明"]
    start = text.find(markers[0])
    if start >= 0:
        text = text[start + len(markers[0]) :].strip()

    for marker in markers[1:]:
        pos = text.find(marker)
        if pos >= 0:
            text = text[:pos].strip()
            break

    if not text:
        text = _normalize_text(answer_markdown)

    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _top_evidence_titles(evidence_items: list[EvidenceItem], top_k: int = 3) -> list[str]:
    """
    获取前若干条证据标题。
    """
    titles: list[str] = []
    for item in evidence_items[:top_k]:
        title = _normalize_text(item.title) or "未命名证据"
        source_file = _normalize_text(item.source_file)
        score = _safe_float(item.score, 0.0)

        if source_file:
            titles.append(f"{title}（{source_file} / score={score:.4f}）")
        else:
            titles.append(f"{title}（score={score:.4f}）")

    return titles


def _has_image_evidence(answer: ChatAnswer) -> bool:
    """
    判断本次回答是否命中了图片证据。
    """
    if answer.gallery_images:
        return True

    for item in answer.evidence_items:
        if item.image_paths:
            return True

    return False


def _build_debug_summary(debug_info: dict[str, Any]) -> dict[str, Any]:
    """
    从 debug_info 中提取简表，避免终端输出过长。
    """
    if not isinstance(debug_info, dict):
        return {"status": "invalid_debug_info"}

    top_results = debug_info.get("top_results", [])
    compact_top_results: list[dict[str, Any]] = []

    if isinstance(top_results, list):
        for item in top_results[:3]:
            if not isinstance(item, dict):
                continue
            compact_top_results.append(
                {
                    "title": _normalize_text(item.get("title")),
                    "score": round(_safe_float(item.get("score"), 0.0), 4),
                    "source_type": _normalize_text(item.get("source_type")),
                    "reason": _normalize_text(item.get("reason")),
                }
            )

    summary = {
        "normalized_query": _normalize_text(debug_info.get("normalized_query")),
        "used_conservative_answer": bool(debug_info.get("used_conservative_answer", False)),
        "llm_called": bool(debug_info.get("llm_called", False)),
        "llm_succeeded": bool(debug_info.get("llm_succeeded", False)),
        "error": _normalize_text(debug_info.get("error")),
        "top_results": compact_top_results,
    }
    return summary


def _classify_case_result(answer: ChatAnswer) -> tuple[str, str]:
    """
    对单条测试结果做粗分类，便于快速回归观察。

    返回：
        (level, description)

    level 可能取值：
        - OK
        - WARN
        - ERROR
    """
    debug_info = answer.debug_info if isinstance(answer.debug_info, dict) else {}
    error_text = _normalize_text(debug_info.get("error"))
    evidence_count = len(answer.evidence_items)
    answer_text = _normalize_text(answer.answer_markdown)
    conservative = bool(debug_info.get("used_conservative_answer", False))
    llm_called = bool(debug_info.get("llm_called", False))
    llm_succeeded = bool(debug_info.get("llm_succeeded", False))

    if not answer_text:
        return "ERROR", "返回答案为空。"

    if error_text:
        return "WARN", f"存在错误信息：{error_text}"

    if evidence_count == 0:
        return "WARN", "未命中任何证据。"

    if conservative and not llm_succeeded:
        if llm_called:
            return "WARN", "触发保守回答，且模型调用未成功。"
        return "WARN", "触发保守回答，说明证据可信度不足。"

    return "OK", "流程正常。"


def _print_case_result(
        *,
        index: int,
        query: str,
        answer: ChatAnswer,
        show_full_answer: bool,
) -> tuple[str, str]:
    """
    打印单个测试问题的结果。
    """
    level, desc = _classify_case_result(answer)
    summary = _build_answer_summary(answer.answer_markdown)
    evidence_titles = _top_evidence_titles(answer.evidence_items, top_k=3)
    has_images = _has_image_evidence(answer)
    debug_summary = _build_debug_summary(answer.debug_info)

    print(SEPARATOR)
    print(f"[测试 {index}] {query}")
    print(SUB_SEPARATOR)
    print(f"结果级别：{level}")
    print(f"结果说明：{desc}")
    print(f"最终答案摘要：{summary}")
    print(f"是否命中图片证据：{'是' if has_images else '否'}")

    if evidence_titles:
        print("Top 3 证据标题：")
        for idx, title in enumerate(evidence_titles, start=1):
            print(f"  {idx}. {title}")
    else:
        print("Top 3 证据标题：未命中证据。")

    print("debug_info 简表：")
    print(debug_summary)

    if show_full_answer:
        print(SUB_SEPARATOR)
        print("完整答案：")
        print(answer.answer_markdown)

    return level, desc


def _run_single_case(
        chat_service: ChatService,
        query: str,
) -> ChatAnswer:
    """
    执行单条冒烟测试。
    """
    return chat_service.chat(query=query, history=None)


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。
    """
    parser = argparse.ArgumentParser(
        description="企业网银 FAQ RAG demo 最小冒烟测试脚本。"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="日志级别，例如 DEBUG / INFO / WARNING / ERROR",
    )
    parser.add_argument(
        "--show-full-answer",
        action="store_true",
        help="是否打印每个测试问题的完整答案正文。",
    )
    return parser.parse_args()


def main() -> int:
    """
    主入口。
    """
    args = parse_args()
    setup_logging(args.log_level, module_name="__main__")

    print(SEPARATOR)
    print("企业网银 FAQ RAG demo 最小冒烟测试")
    print(SEPARATOR)

    try:
        chat_service = ChatService()
    except ChatServiceError as exc:
        logger.exception("ChatService 初始化失败：%s", exc)
        print("【严重错误】ChatService 初始化失败，无法执行冒烟测试。")
        print(f"详细信息：{exc}")
        return 1
    except Exception as exc:
        logger.exception("系统初始化出现未预期异常：%s", exc)
        print("【严重错误】系统初始化出现未预期异常，无法执行冒烟测试。")
        print(f"详细信息：{exc}")
        return 1

    ok_count = 0
    warn_count = 0
    error_count = 0

    for idx, query in enumerate(DEFAULT_TEST_QUERIES, start=1):
        try:
            answer = _run_single_case(chat_service, query)
            level, _ = _print_case_result(
                index=idx,
                query=query,
                answer=answer,
                show_full_answer=args.show_full_answer,
            )

            if level == "OK":
                ok_count += 1
            elif level == "WARN":
                warn_count += 1
            else:
                error_count += 1

        except Exception as exc:
            logger.exception("测试问题执行失败：query=%s, err=%s", query, exc)
            print(SEPARATOR)
            print(f"[测试 {idx}] {query}")
            print(SUB_SEPARATOR)
            print("结果级别：ERROR")
            print("结果说明：执行过程中发生未捕获异常。")
            print(f"异常详情：{exc}")
            print("提示：这通常意味着 API 调用、索引加载或服务层逻辑存在问题。")
            error_count += 1

    print(SEPARATOR)
    print("冒烟测试汇总")
    print(SUB_SEPARATOR)
    print(f"总问题数：{len(DEFAULT_TEST_QUERIES)}")
    print(f"OK：{ok_count}")
    print(f"WARN：{warn_count}")
    print(f"ERROR：{error_count}")

    if error_count > 0:
        print("结论：存在明显异常，建议优先检查日志、debug_info、索引文件和 API 配置。")
        return 1

    if warn_count > 0:
        print("结论：主流程可运行，但存在证据不足、保守回答或局部能力退化现象。")
        return 0

    print("结论：主流程运行正常，冒烟测试通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())