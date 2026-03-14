from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import ValidationError

# 兼容直接执行：
# python scripts/parse_excel_faq.py --input ... --output ...
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.logging_utils import setup_logging  # noqa: E402
from app.models import KBChunk  # noqa: E402

import logging

logger = logging.getLogger(__name__)

HEADER_ALIASES: dict[str, tuple[str, ...]] = {
    "序号": ("序号",),
    "功能大类": ("功能大类", "功能分类", "业务大类"),
    "问题描述": ("问题描述", "问题", "常见问题", "问题说明"),
    "解决字段": ("解决方法", "解决办法", "处理办法", "处理方式", "解决方案"),
}
WHITESPACE_RE = re.compile(r"\s+")
DELIMITER_RE = re.compile(r"[，,。；;：:、/\\|（）()\[\]【】《》<>\-—_]+")
QUESTION_NOISE_RE = re.compile(
    r"(怎么办|如何处理|怎么处理|如何|怎么|为何|为什么|请问|提示|报错|失败|异常|无法|不能|不可以|是否可以)"
)


def normalize_text(value: Any) -> str:
    """
    将单元格值清洗为稳定字符串：
    1. 空值转空字符串
    2. 去除首尾空白
    3. 连续空白折叠为单个空格
    """
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except TypeError:
        # 某些对象类型不支持 pd.isna，忽略即可
        pass

    text = str(value).replace("\u3000", " ")
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def normalize_header_name(header: Any) -> str:
    """
    归一化表头名称，便于兼容表头中夹杂空格、换行等情况。
    """
    return normalize_text(header).replace(" ", "")


def build_header_mapping(columns: list[str]) -> dict[str, str]:
    """
    根据 Excel 实际表头构建“标准字段 -> 实际列名”的映射。
    支持表头别名，例如“解决方法 / 解决办法 / 解决方案”。
    """
    normalized_to_raw: dict[str, str] = {}
    for col in columns:
        normalized_to_raw[normalize_header_name(col)] = col

    mapping: dict[str, str] = {}
    missing: list[str] = []

    for standard_name, aliases in HEADER_ALIASES.items():
        matched_raw: str | None = None
        for alias in aliases:
            alias_key = normalize_header_name(alias)
            raw = normalized_to_raw.get(alias_key)
            if raw is not None:
                matched_raw = raw
                break

        if matched_raw is None:
            missing.append(f"{standard_name} <- {list(aliases)}")
        else:
            mapping[standard_name] = matched_raw

    if missing:
        raise ValueError(
            f"Excel 缺少必要表头映射: {missing}。"
            f"实际表头为: {columns}。"
        )

    return mapping


def make_title(question: str, max_length: int = 24) -> str:
    """
    基于“问题描述”生成简短标题。
    优先截取第一个语义停顿点之前的内容；若过长则截断。
    """
    question = normalize_text(question)
    if not question:
        return "未命名FAQ"

    # 优先按常见中文停顿符截取
    first_part = re.split(r"[，,。；;：:（(]", question, maxsplit=1)[0].strip()
    candidate = first_part or question

    if len(candidate) <= max_length:
        return candidate

    return candidate[: max_length - 1].rstrip() + "…"


def extract_keywords(category: str, question: str, max_keywords: int = 12) -> list[str]:
    """
    提取基础关键词。
    设计目标是“轻量、稳定、无额外依赖”，因此使用规则法而非分词库。

    提取策略：
    1. 直接保留分类名
    2. 对问题文本做基础去噪
    3. 依据标点切分短语
    4. 补充英文/数字/UKey 之类的字母数字串
    """
    candidates: list[str] = []

    category = normalize_text(category)
    question = normalize_text(question)

    if category:
        candidates.append(category)

    cleaned_question = QUESTION_NOISE_RE.sub(" ", question)
    phrase_parts = [p.strip() for p in DELIMITER_RE.split(cleaned_question) if p.strip()]
    candidates.extend(phrase_parts)

    # 提取字母数字串，例如 UKey、USB、IE、EDGE、12306 之类
    alpha_num_tokens = re.findall(r"[A-Za-z][A-Za-z0-9._-]*|\d+[A-Za-z0-9._-]*", question)
    candidates.extend(alpha_num_tokens)

    results: list[str] = []
    seen: set[str] = set()

    for item in candidates:
        item = normalize_text(item)
        if not item:
            continue
        # 过滤过长或过短的噪声项
        if len(item) < 2:
            continue
        if len(item) > 24:
            continue
        lower_key = item.casefold()
        if lower_key in seen:
            continue
        seen.add(lower_key)
        results.append(item)
        if len(results) >= max_keywords:
            break

    return results


def build_full_text(category: str, question: str, answer: str) -> str:
    """
    构造适合 embedding 的自然语言文本。
    """
    category = normalize_text(category)
    question = normalize_text(question)
    answer = normalize_text(answer)

    parts: list[str] = []
    if category:
        parts.append(f"功能分类：{category}")
    if question:
        parts.append(f"常见问题：{question}")
    if answer:
        parts.append(f"参考解答：{answer}")

    return "。".join(parts).strip("。") + "。"


def compute_chunk_hash(
        source_file: str,
        category: str,
        question: str,
        answer: str,
) -> str:
    """
    生成稳定的知识块哈希，便于去重与追踪。
    """
    raw = f"{source_file}||{normalize_text(category)}||{normalize_text(question)}||{normalize_text(answer)}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def build_kb_chunk(
        *,
        source_path: Path,
        row_index: int,
        seq_no: str,
        category: str,
        question: str,
        answer: str,
) -> KBChunk:
    """
    将单行 FAQ 转换为统一的 KBChunk。
    """
    source_file = source_path.name
    title = make_title(question)
    full_text = build_full_text(category, question, answer)
    keywords = extract_keywords(category, question)
    chunk_hash = compute_chunk_hash(source_file, category, question, answer)

    # 若 Excel “序号”为空，则退化为行号
    suffix = seq_no or str(row_index)

    return KBChunk(
        doc_id=f"{source_path.stem}-{suffix}",
        source_file=source_file,
        source_type="excel",
        title=title,
        category=category or None,
        question=question or None,
        answer=answer or None,
        full_text=full_text,
        keywords=keywords,
        image_paths=[],
        page_no=None,
        slide_no=None,
        priority=1.0,
        chunk_hash=chunk_hash,
    )


def read_excel_faq(input_path: Path, sheet_name: str | int | None = 0) -> pd.DataFrame:
    """
    读取 FAQ Excel 文件并返回 DataFrame。
    """
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在：{input_path}")

    if not input_path.is_file():
        raise ValueError(f"输入路径不是文件：{input_path}")

    suffix = input_path.suffix.lower()
    if suffix not in {".xlsx", ".xls", ".xlsm"}:
        raise ValueError(f"暂不支持的 Excel 文件类型：{suffix}")

    try:
        df = pd.read_excel(
            input_path,
            sheet_name=sheet_name,
            dtype=object,
            engine=None,  # 让 pandas 自动选择，常见场景会使用 openpyxl
        )
    except Exception as exc:
        raise RuntimeError(f"读取 Excel 失败：{input_path}") from exc

    if not isinstance(df, pd.DataFrame):
        raise RuntimeError("读取结果不是 DataFrame，请检查 sheet_name 是否正确。")

    if df.empty:
        logger.warning("Excel 文件为空：%s", input_path)

    # 统一清理列名显示形式
    df.columns = [normalize_text(col) for col in df.columns]
    logger.info("Excel 识别到的表头：%s", list(df.columns))
    return df


def parse_excel_faq(input_path: Path, sheet_name: str | int | None = 0) -> list[KBChunk]:
    """
    解析企业网银 FAQ Excel，并返回结构化知识块列表。
    """
    df = read_excel_faq(input_path=input_path, sheet_name=sheet_name)
    header_mapping = build_header_mapping(list(df.columns))

    seq_col = header_mapping["序号"]
    category_col = header_mapping["功能大类"]
    question_col = header_mapping["问题描述"]
    answer_col = header_mapping["解决字段"]

    chunks: list[KBChunk] = []
    skipped_count = 0

    for idx, row in df.iterrows():
        excel_row_no = idx + 2  # DataFrame 第 0 行对应 Excel 第 2 行（第 1 行通常是表头）

        seq_no = normalize_text(row.get(seq_col, ""))
        category = normalize_text(row.get(category_col, ""))
        question = normalize_text(row.get(question_col, ""))
        answer = normalize_text(row.get(answer_col, ""))

        # 全空行直接跳过
        if not any([seq_no, category, question, answer]):
            skipped_count += 1
            logger.debug("跳过空行：Excel 第 %s 行", excel_row_no)
            continue

        # FAQ 至少需要问题或答案之一；两者都空则没有实际意义
        if not question and not answer:
            skipped_count += 1
            logger.warning("跳过无有效问答内容的行：Excel 第 %s 行", excel_row_no)
            continue

        try:
            chunk = build_kb_chunk(
                source_path=input_path,
                row_index=excel_row_no,
                seq_no=seq_no,
                category=category,
                question=question,
                answer=answer,
            )
            chunks.append(chunk)
        except ValidationError as exc:
            skipped_count += 1
            logger.warning("KBChunk 校验失败，已跳过 Excel 第 %s 行：%s", excel_row_no, exc)
        except Exception as exc:
            skipped_count += 1
            logger.exception("处理 Excel 第 %s 行时发生异常：%s", excel_row_no, exc)

    logger.info(
        "Excel 解析完成：总行数=%s，成功=%s，跳过=%s，文件=%s",
        len(df),
        len(chunks),
        skipped_count,
        input_path,
    )
    return chunks


def save_chunks_to_jsonl(chunks: list[KBChunk], output_path: Path) -> None:
    """
    将知识块保存为 JSONL。
    每行一个 KBChunk，适合后续索引与调试。
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with output_path.open("w", encoding="utf-8") as f:
            for chunk in chunks:
                line = json.dumps(
                    chunk.model_dump(mode="json", exclude_none=False),
                    ensure_ascii=False,
                )
                f.write(line + "\n")
    except OSError as exc:
        raise RuntimeError(f"写入 JSONL 失败：{output_path}") from exc

    logger.info("JSONL 已保存：%s（共 %s 条）", output_path, len(chunks))


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。
    """
    parser = argparse.ArgumentParser(
        description="解析企业网银 FAQ Excel 文件，并输出结构化 KBChunk JSONL。"
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="输入 Excel 文件路径，例如 data/raw/faq.xlsx",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="输出 JSONL 文件路径，例如 data/processed/faq_chunks.jsonl",
    )
    parser.add_argument(
        "--sheet",
        default=0,
        help="要读取的工作表名称或索引，默认 0",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="日志级别，例如 DEBUG / INFO / WARNING / ERROR",
    )
    return parser.parse_args()


def parse_sheet_arg(value: Any) -> str | int | None:
    """
    将 --sheet 参数转换为 pandas 可接受的 sheet_name。
    - 若是纯数字字符串，则转为 int
    - 否则按工作表名称处理
    """
    if value is None:
        return 0

    text = normalize_text(value)
    if not text:
        return 0

    if text.isdigit():
        return int(text)

    return text


def main() -> int:
    """
    命令行主入口。
    """
    args = parse_args()
    setup_logging(args.log_level)

    input_path: Path = args.input.expanduser().resolve()
    output_path: Path = args.output.expanduser().resolve()
    sheet_name = parse_sheet_arg(args.sheet)

    logger.info("开始解析 Excel FAQ：input=%s, output=%s, sheet=%s", input_path, output_path, sheet_name)

    try:
        chunks = parse_excel_faq(input_path=input_path, sheet_name=sheet_name)
        save_chunks_to_jsonl(chunks=chunks, output_path=output_path)
        return 0
    except Exception as exc:
        logger.exception("执行失败：%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())