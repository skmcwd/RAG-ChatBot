from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd

from app.config import get_settings
from app.models import ChatAnswer, EvidenceItem
from app.services.chat_service import ChatService

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

CUSTOM_CSS = """
:root {
  --ebank-primary: #0f4c81;
  --ebank-accent: #1f6feb;
  --ebank-bg: #f6f8fb;
  --ebank-card: #ffffff;
  --ebank-border: #dce3ec;
  --ebank-text: #16324f;
  --ebank-muted: #6b7785;
}

.gradio-container {
  background: linear-gradient(180deg, #f7f9fc 0%, #f3f6fb 100%);
}

#ebank-header {
  background: linear-gradient(135deg, rgba(15,76,129,0.08), rgba(31,111,235,0.06));
  border: 1px solid var(--ebank-border);
  border-radius: 18px;
  padding: 18px 22px;
  margin-bottom: 14px;
}

#ebank-header h1 {
  margin: 0 0 6px 0;
  color: var(--ebank-primary);
  font-size: 28px;
  font-weight: 700;
}

#ebank-header p {
  margin: 0;
  color: var(--ebank-muted);
  font-size: 14px;
}

.ebank-panel {
  background: var(--ebank-card);
  border: 1px solid var(--ebank-border);
  border-radius: 18px;
  padding: 14px;
  box-shadow: 0 6px 24px rgba(15, 76, 129, 0.06);
}

.ebank-subtitle {
  color: var(--ebank-text);
  font-size: 15px;
  font-weight: 600;
  margin-bottom: 8px;
}

.ebank-footnote {
  color: var(--ebank-muted);
  font-size: 12px;
  margin-top: 6px;
}
"""

EVIDENCE_TABLE_COLUMNS = [
    "doc_id",
    "标题",
    "来源文件",
    "来源类型",
    "分类",
    "位置",
    "分数",
    "命中原因",
]

DEFAULT_EVIDENCE_SUMMARY = """### 来源依据
当前尚未产生回答。发送问题后，这里将显示本次回答的证据摘要与命中情况。"""

DEFAULT_DEBUG_INFO = {
    "status": "idle",
    "message": "等待用户提问。",
}


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
    稳健转 float。
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _unique_keep_order(items: list[str]) -> list[str]:
    """
    保持原顺序去重。
    """
    result: list[str] = []
    seen: set[str] = set()

    for item in items:
        text = _normalize_text(item)
        if not text:
            continue

        key = text.casefold()
        if key in seen:
            continue

        seen.add(key)
        result.append(text)

    return result


def _empty_evidence_dataframe() -> pd.DataFrame:
    """
    返回空证据表。
    """
    return pd.DataFrame(columns=EVIDENCE_TABLE_COLUMNS)


def _resolve_media_path(path_str: str) -> str | None:
    """
    将相对路径或绝对路径解析为可供 Gradio Gallery 使用的本地文件路径。
    """
    clean = _normalize_text(path_str)
    if not clean:
        return None

    path = Path(clean)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()

    if not path.exists() or not path.is_file():
        return None

    return str(path)


def _format_location(item: EvidenceItem) -> str:
    """
    格式化 slide/page 位置信息。
    """
    if item.slide_no is not None:
        return f"第 {item.slide_no} 页/张"
    if item.page_no is not None:
        return f"第 {item.page_no} 页"
    return "-"


def _build_evidence_summary(answer: ChatAnswer) -> str:
    """
    构造右侧“证据摘要” Markdown。
    """
    evidence_items = answer.evidence_items or []
    if not evidence_items:
        return """### 来源依据
未检索到可展示的证据。"""

    lines: list[str] = [
        "### 来源依据",
        f"本次回答共参考 **{len(evidence_items)}** 条证据，以下为主要命中内容：",
        "",
    ]

    for idx, item in enumerate(evidence_items[:5], start=1):
        title = _normalize_text(item.title) or "未命名资料"
        source_file = _normalize_text(item.source_file) or "未知文件"
        source_type = _normalize_text(item.source_type) or "未知类型"
        category = _normalize_text(item.category) or "未分类"
        location = _format_location(item)
        score = _safe_float(item.score, 0.0)
        reason = _normalize_text(item.reason) or "无"
        lines.extend(
            [
                f"**证据 {idx}**：{title}",
                f"- 来源：`{source_file}` · `{source_type}` · 分类：`{category}` · 位置：`{location}`",
                f"- 分数：`{score:.4f}`",
                f"- 命中原因：{reason}",
                "",
            ]
        )

    return "\n".join(lines).strip()


def _build_evidence_dataframe(answer: ChatAnswer) -> pd.DataFrame:
    """
    构造右侧证据表格。
    """
    rows: list[list[Any]] = []

    for item in answer.evidence_items or []:
        rows.append(
            [
                _normalize_text(item.doc_id),
                _normalize_text(item.title),
                _normalize_text(item.source_file),
                _normalize_text(item.source_type),
                _normalize_text(item.category) or "未分类",
                _format_location(item),
                round(_safe_float(item.score, 0.0), 6),
                _normalize_text(item.reason),
                ]
        )

    if not rows:
        return _empty_evidence_dataframe()

    return pd.DataFrame(rows, columns=EVIDENCE_TABLE_COLUMNS)


def _build_gallery_items(answer: ChatAnswer) -> list[tuple[str, str | None]]:
    """
    构造 Gallery 输出值。
    """
    gallery_items: list[tuple[str, str | None]] = []

    for idx, path_str in enumerate(answer.gallery_images or [], start=1):
        resolved = _resolve_media_path(path_str)
        if not resolved:
            continue
        gallery_items.append((resolved, f"相关截图 {idx}"))

    return gallery_items


def _build_source_basis_lines(evidence_items: list[EvidenceItem]) -> list[str]:
    """
    构造附加在答案末尾的“来源依据”文本。
    """
    if not evidence_items:
        return ["### 来源依据", "- 当前未检索到可展示的证据。"]

    lines: list[str] = ["### 来源依据"]
    for idx, item in enumerate(evidence_items[:4], start=1):
        title = _normalize_text(item.title) or "未命名资料"
        source_file = _normalize_text(item.source_file) or "未知文件"
        source_type = _normalize_text(item.source_type) or "未知类型"
        category = _normalize_text(item.category) or "未分类"
        location = _format_location(item)
        score = _safe_float(item.score, 0.0)
        lines.append(
            f"- [{idx}] {title}（{source_file} / {source_type} / {category} / {location} / score={score:.4f}）"
        )

    return lines


def _build_chatbot_answer(answer: ChatAnswer) -> str:
    """
    将 ChatAnswer 转为 Chatbot 中展示的最终 Markdown。
    在答案末尾显式补充“来源依据”。
    """
    answer_markdown = _normalize_text(answer.answer_markdown)
    source_lines = _build_source_basis_lines(answer.evidence_items or [])

    if answer_markdown:
        return f"{answer_markdown}\n\n" + "\n".join(source_lines)

    return "\n".join(source_lines)


def _debug_payload(answer: ChatAnswer) -> dict[str, Any]:
    """
    确保 debug_info 为 JSON 友好的字典结构。
    """
    debug_info = answer.debug_info if isinstance(answer.debug_info, dict) else {}
    return debug_info or {"status": "ok", "message": "无额外调试信息。"}


@lru_cache(maxsize=1)
def _get_chat_service() -> ChatService:
    """
    单例化 ChatService，避免重复初始化索引与客户端。
    """
    return ChatService()


def _safe_ui_settings() -> tuple[str, str, list[str]]:
    """
    稳健读取 UI 配置。
    """
    try:
        settings = get_settings()
        title = _normalize_text(settings.ui.app_title) or "企业网银问题助手"
        subtitle = _normalize_text(settings.ui.app_subtitle) or "基于本地知识库的企业网银 FAQ RAG 演示"
        examples = [
            _normalize_text(q)
            for q in settings.ui.example_questions
            if _normalize_text(q)
        ]
        if not examples:
            examples = [
                "企业网银登录时提示控件未安装怎么办？",
                "UKey 插入后无法识别如何处理？",
                "回单在哪里查询和下载？",
            ]
        return title, subtitle, examples
    except Exception as exc:
        logger.warning("读取 UI 配置失败，使用默认值：err=%s", exc)
        return (
            "企业网银问题助手",
            "基于本地知识库的企业网银 FAQ RAG 演示",
            [
                "企业网银登录时提示控件未安装怎么办？",
                "UKey 插入后无法识别如何处理？",
                "回单在哪里查询和下载？",
            ],
        )


def _register_static_paths() -> None:
    """
    注册静态图片目录，便于 Gallery 直接展示本地解析出的截图。
    """
    try:
        settings = get_settings()
        image_dir = settings.paths.parsed_images_dir
        if image_dir.exists() and image_dir.is_dir():
            gr.set_static_paths(paths=[image_dir])
            logger.info("已注册 Gradio 静态目录：%s", image_dir)
    except Exception as exc:
        logger.warning("注册静态目录失败，将继续运行：err=%s", exc)


def _fill_example(example_text: str) -> str:
    """
    点击示例问题按钮时，将内容填入输入框。
    """
    return _normalize_text(example_text)


def _handle_chat(
        query: str,
        chat_state: list[dict[str, str]] | None,
) -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
    str,
    str,
    pd.DataFrame,
    list[tuple[str, str | None]],
    dict[str, Any],
]:
    """
    单轮对话处理函数。

    返回顺序必须与事件 outputs 严格一致：
    1. chat_state
    2. chatbot
    3. query_box
    4. evidence_summary
    5. evidence_table
    6. gallery
    7. debug_json
    """
    history = list(chat_state or [])
    clean_query = _normalize_text(query)

    if not clean_query:
        gr.Warning("请输入问题后再发送。")
        return (
            history,
            history,
            "",
            DEFAULT_EVIDENCE_SUMMARY,
            _empty_evidence_dataframe(),
            [],
            {
                "status": "warning",
                "message": "empty_query",
            },
        )

    try:
        service = _get_chat_service()
        answer = service.chat(clean_query, history=history)

        user_message = {"role": "user", "content": clean_query}
        assistant_message = {
            "role": "assistant",
            "content": _build_chatbot_answer(answer),
        }

        new_history = [*history, user_message, assistant_message]

        evidence_summary = _build_evidence_summary(answer)
        evidence_df = _build_evidence_dataframe(answer)
        gallery_items = _build_gallery_items(answer)
        debug_info = _debug_payload(answer)

        return (
            new_history,
            new_history,
            "",
            evidence_summary,
            evidence_df,
            gallery_items,
            debug_info,
        )

    except Exception as exc:
        logger.exception("UI 对话处理失败：%s", exc)
        gr.Warning("本次请求处理失败，请查看调试信息或稍后重试。")

        error_text = (
            "一、结论\n"
            "根据当前知识库无法确认。\n\n"
            "二、操作步骤\n"
            "请稍后重试，或缩小问题范围并补充更具体的报错原文、截图、菜单路径、错误码。\n\n"
            "三、补充说明\n"
            "当前请求在服务执行过程中发生异常，未生成可信答案。\n\n"
            "### 来源依据\n"
            "- 本次请求因系统异常未完成证据整理。"
        )

        new_history = [
            *history,
            {"role": "user", "content": clean_query},
            {"role": "assistant", "content": error_text},
        ]

        debug_info = {
            "status": "error",
            "message": str(exc),
        }

        return (
            new_history,
            new_history,
            "",
            "### 来源依据\n当前请求处理失败，未生成有效证据摘要。",
            _empty_evidence_dataframe(),
            [],
            debug_info,
        )


def _clear_all() -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
    str,
    str,
    pd.DataFrame,
    list[tuple[str, str | None]],
    dict[str, Any],
]:
    """
    清空会话与右侧展示区域。
    """
    empty_history: list[dict[str, str]] = []
    return (
        empty_history,
        empty_history,
        "",
        DEFAULT_EVIDENCE_SUMMARY,
        _empty_evidence_dataframe(),
        [],
        DEFAULT_DEBUG_INFO,
    )


def build_demo() -> gr.Blocks:
    """
    构建 Gradio Blocks 界面。
    """
    _register_static_paths()
    app_title, app_subtitle, example_questions = _safe_ui_settings()

    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
    )

    with gr.Blocks(
            title=app_title,
            theme=theme,
            css=CUSTOM_CSS,
            fill_height=True,
    ) as demo:
        chat_state = gr.State(value=[])

        gr.HTML(
            f"""
            <div id="ebank-header">
              <h1>企业网银问题助手</h1>
              <p>{app_subtitle}</p>
            </div>
            """
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=7, min_width=720):
                with gr.Group(elem_classes=["ebank-panel"]):
                    gr.Markdown("#### 对话区", elem_classes=["ebank-subtitle"])

                    chatbot = gr.Chatbot(
                        value=[],
                        type="messages",
                        height=560,
                        show_copy_button=True,
                        bubble_full_width=False,
                        placeholder="请输入企业网银相关问题，例如登录、UKey、证书、回单、转账、代发、权限等。",
                        label="问题对话",
                    )

                    query_box = gr.Textbox(
                        value="",
                        label="请输入你的问题",
                        placeholder="例如：企业网银登录时提示安全控件异常怎么办？",
                        lines=3,
                        max_lines=6,
                        autofocus=True,
                    )

                    with gr.Row():
                        send_btn = gr.Button("发送", variant="primary", size="lg")
                        clear_btn = gr.Button("清空历史", variant="secondary", size="lg")

                    gr.Markdown("#### 示例问题", elem_classes=["ebank-subtitle"])
                    with gr.Group():
                        for i in range(0, len(example_questions), 2):
                            with gr.Row():
                                for example in example_questions[i : i + 2]:
                                    example_btn = gr.Button(
                                        value=example,
                                        variant="secondary",
                                        size="sm",
                                    )
                                    example_btn.click(
                                        fn=lambda text=example: _fill_example(text),
                                        inputs=None,
                                        outputs=query_box,
                                    )

                    gr.Markdown(
                        "提示：答案下方将显式展示“来源依据”，便于会议、演示与客户交流时核对。",
                        elem_classes=["ebank-footnote"],
                    )

            with gr.Column(scale=5, min_width=520):
                with gr.Group(elem_classes=["ebank-panel"]):
                    gr.Markdown("#### 证据摘要", elem_classes=["ebank-subtitle"])
                    evidence_summary = gr.Markdown(
                        value=DEFAULT_EVIDENCE_SUMMARY,
                        label="证据摘要",
                    )

                with gr.Group(elem_classes=["ebank-panel"]):
                    gr.Markdown("#### 命中证据表", elem_classes=["ebank-subtitle"])
                    evidence_table = gr.Dataframe(
                        value=_empty_evidence_dataframe(),
                        headers=EVIDENCE_TABLE_COLUMNS,
                        datatype=["str", "str", "str", "str", "str", "str", "number", "str"],
                        interactive=False,
                        wrap=True,
                        row_count=(0, "dynamic"),
                        col_count=(len(EVIDENCE_TABLE_COLUMNS), "fixed"),
                        height=260,
                        label="命中证据",
                    )

                with gr.Group(elem_classes=["ebank-panel"]):
                    gr.Markdown("#### 相关截图", elem_classes=["ebank-subtitle"])
                    gallery = gr.Gallery(
                        value=[],
                        label="相关截图",
                        show_label=False,
                        columns=2,
                        height=260,
                        preview=True,
                        object_fit="contain",
                    )

                with gr.Accordion("调试信息", open=False):
                    debug_json = gr.JSON(
                        value=DEFAULT_DEBUG_INFO,
                        label="debug_info",
                    )

        send_outputs = [
            chat_state,
            chatbot,
            query_box,
            evidence_summary,
            evidence_table,
            gallery,
            debug_json,
        ]

        send_btn.click(
            fn=_handle_chat,
            inputs=[query_box, chat_state],
            outputs=send_outputs,
        )

        query_box.submit(
            fn=_handle_chat,
            inputs=[query_box, chat_state],
            outputs=send_outputs,
        )

        clear_btn.click(
            fn=_clear_all,
            inputs=None,
            outputs=send_outputs,
        )

    return demo


demo = build_demo()


if __name__ == "__main__":
    demo.launch()