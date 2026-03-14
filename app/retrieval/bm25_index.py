from __future__ import annotations

import json
import logging
import pickle
import re
# import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError
from rank_bm25 import BM25Okapi

from app.config import get_settings
from app.retrieval.query_normalizer import normalize_query

logger = logging.getLogger(__name__)

WHITESPACE_RE = re.compile(r"\s+")
PUNCT_SPLIT_RE = re.compile(r"[，,。；;：:、/\\|（）()\[\]【】《》<>\-—_~!！?？\"'“”‘’\s]+")
ASCII_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
CJK_SPAN_RE = re.compile(r"[\u4e00-\u9fff]+")
ERROR_CODE_RE = re.compile(r"\b(?:[A-Za-z]{1,8}[-_ ]?\d{2,8}|0x[0-9A-Fa-f]{3,16})\b")
ENGLISH_PHRASE_RE = re.compile(r"\b[a-zA-Z]{2,}(?:\s+[a-zA-Z]{2,}){1,5}\b")
MENU_PATH_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9_]+(?:\s*(?:>|/|->|→)\s*[\u4e00-\u9fffA-Za-z0-9_]+){1,8}")

DEFAULT_TOP_K = 5
DEFAULT_EXACT_TERM_BOOST = 1.2
DEFAULT_SPECIAL_EXACT_BOOST = 1.8
DEFAULT_CATEGORY_BOOST = 0.35
DEFAULT_PRIORITY_WEIGHT = 0.10


class BM25IndexError(RuntimeError):
    """BM25 索引加载或检索异常。"""


class BM25Document(BaseModel):
    """
    本地 BM25 语料中的单条记录。

    该结构与 scripts/build_indexes.py 中保存的 bm25_corpus.jsonl 保持兼容。
    """

    doc_id: str
    chunk_hash: str = ""
    bm25_text: str = ""
    tokens: list[str] = Field(default_factory=list)

    source_file: str = ""
    source_type: str = ""
    title: str = ""
    category: str | None = None
    priority: float = 1.0
    image_paths: list[str] = Field(default_factory=list)
    slide_no: int | None = None
    page_no: int | None = None


class BM25SearchResult(BaseModel):
    """
    BM25 检索结果。

    该结构可直接用于后续 hybrid merge。
    """

    doc_id: str
    chunk_hash: str = ""
    retrieval_score: float = 0.0
    bm25_score: float = 0.0
    exact_match_boost: float = 0.0
    category_boost: float = 0.0
    priority_boost: float = 0.0

    source_file: str = ""
    source_type: str = ""
    title: str = ""
    category: str | None = None
    priority: float = 1.0
    image_paths: list[str] = Field(default_factory=list)
    slide_no: int | None = None
    page_no: int | None = None

    matched_terms: list[str] = Field(default_factory=list)
    exact_terms: list[str] = Field(default_factory=list)
    query_normalized: str = ""
    snippet: str = ""


def _normalize_text(value: Any) -> str:
    """
    统一文本清洗：
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


def _tokenize_for_bm25(text: str) -> list[str]:
    """
    轻量中英混合分词。
    与 scripts/build_indexes.py 的 tokenization 逻辑保持一致。

    规则：
    1. 英文/数字 token 直接保留
    2. 中文连续片段加入短整段 + 2-gram + 3-gram
    3. 标点切分后的普通片段作为补充
    """
    normalized = _normalize_text(text).lower()
    if not normalized:
        return []

    tokens: list[str] = []

    ascii_tokens = ASCII_TOKEN_RE.findall(normalized)
    tokens.extend(ascii_tokens)

    for span in CJK_SPAN_RE.findall(normalized):
        if not span:
            continue

        if len(span) <= 8:
            tokens.append(span)

        if len(span) >= 2:
            tokens.extend(span[i : i + 2] for i in range(len(span) - 1))

        if len(span) >= 3:
            tokens.extend(span[i : i + 3] for i in range(len(span) - 2))

    for part in PUNCT_SPLIT_RE.split(normalized):
        part = _normalize_text(part)
        if 2 <= len(part) <= 24:
            tokens.append(part)

    return [token for token in (_normalize_text(t) for t in tokens) if token]


def _safe_float(value: Any, default: float = 0.0) -> float:
    """
    将任意值稳健转为 float。
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract_special_exact_terms(query: str) -> list[str]:
    """
    从原始 query 中提取需要优先精确命中的术语。

    重点覆盖：
    1. 错误码 / 十六进制码
    2. 英文报错短语
    3. 菜单路径关键词
    """
    query_text = _normalize_text(query)
    if not query_text:
        return []

    exact_terms: list[str] = []

    exact_terms.extend(ERROR_CODE_RE.findall(query_text))
    exact_terms.extend(MENU_PATH_RE.findall(query_text))

    for phrase in ENGLISH_PHRASE_RE.findall(query_text):
        phrase_text = _normalize_text(phrase)
        if len(phrase_text.split()) >= 2:
            exact_terms.append(phrase_text)

    return _unique_keep_order(exact_terms)


def _build_query_text(normalized_query: str, expanded_terms: list[str]) -> str:
    """
    将归一化问题与扩展词合成为 BM25 查询串。
    """
    parts = [normalized_query, *expanded_terms]
    return " ".join(part for part in (_normalize_text(p) for p in parts) if part)


def _make_searchable_text(doc: BM25Document) -> str:
    """
    构造用于 exact term 匹配的检索文本。
    """
    parts = [
        doc.title,
        doc.category or "",
        doc.bm25_text,
        " ".join(doc.tokens),
        ]
    return " | ".join(part for part in (_normalize_text(p) for p in parts) if part).casefold()


def _count_exact_matches(
        searchable_text: str,
        exact_terms: list[str],
) -> tuple[list[str], int]:
    """
    统计精确术语命中情况。
    """
    matched: list[str] = []
    score_count = 0

    for term in exact_terms:
        clean_term = _normalize_text(term)
        if not clean_term:
            continue

        if clean_term.casefold() in searchable_text:
            matched.append(clean_term)
            score_count += 1

    return _unique_keep_order(matched), score_count


def _match_general_terms(
        searchable_text: str,
        terms: list[str],
) -> list[str]:
    """
    统计普通命中词。
    """
    matched: list[str] = []
    for term in terms:
        clean_term = _normalize_text(term)
        if not clean_term:
            continue

        if clean_term.casefold() in searchable_text:
            matched.append(clean_term)

    return _unique_keep_order(matched)


class BM25Index:
    """
    本地 BM25 索引加载与查询类。

    功能：
    1. 从 data/index/bm25/ 加载离线构建好的语料与索引
    2. 支持 query_normalizer 的扩展词联合检索
    3. 支持 exact term boost
    4. 返回结构化结果，便于后续 hybrid merge
    """

    def __init__(
            self,
            bm25_dir: Path | None = None,
            *,
            exact_term_boost: float = DEFAULT_EXACT_TERM_BOOST,
            special_exact_boost: float = DEFAULT_SPECIAL_EXACT_BOOST,
            category_boost: float = DEFAULT_CATEGORY_BOOST,
            priority_weight: float = DEFAULT_PRIORITY_WEIGHT,
    ) -> None:
        settings = get_settings()
        self.bm25_dir = (bm25_dir or settings.paths.bm25_dir).expanduser().resolve()

        self.corpus_path = self.bm25_dir / "bm25_corpus.jsonl"
        self.index_path = self.bm25_dir / "bm25_index.pkl"
        self.meta_path = self.bm25_dir / "bm25_meta.json"

        self.exact_term_boost = float(exact_term_boost)
        self.special_exact_boost = float(special_exact_boost)
        self.category_boost = float(category_boost)
        self.priority_weight = float(priority_weight)

        self._documents: list[BM25Document] = []
        self._doc_id_to_index: dict[str, int] = {}
        self._bm25: BM25Okapi | None = None
        self._meta: dict[str, Any] = {}

    @property
    def documents(self) -> list[BM25Document]:
        """
        已加载的 BM25 文档列表。
        """
        return self._documents

    @property
    def doc_id_to_index(self) -> dict[str, int]:
        """
        doc_id 到文档下标的映射。
        """
        return self._doc_id_to_index

    @property
    def meta(self) -> dict[str, Any]:
        """
        索引元信息。
        """
        return self._meta

    def is_loaded(self) -> bool:
        """
        判断索引是否已加载。
        """
        return self._bm25 is not None and bool(self._documents)

    def load(self) -> None:
        """
        从本地目录加载 BM25 索引。

        加载内容：
        1. bm25_corpus.jsonl：tokenized corpus + doc_id 映射 + metadata
        2. bm25_index.pkl：预构建 BM25Okapi 对象；若不可用则自动重建
        3. bm25_meta.json：可选元信息
        """
        if not self.corpus_path.exists():
            raise BM25IndexError(f"未找到 BM25 语料文件：{self.corpus_path}")

        if not self.corpus_path.is_file():
            raise BM25IndexError(f"BM25 语料路径不是文件：{self.corpus_path}")

        documents: list[BM25Document] = []
        doc_id_to_index: dict[str, int] = {}

        try:
            with self.corpus_path.open("r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    text = line.strip()
                    if not text:
                        continue

                    try:
                        obj = json.loads(text)
                        doc = BM25Document.model_validate(obj)
                    except (json.JSONDecodeError, ValidationError) as exc:
                        raise BM25IndexError(
                            f"BM25 语料解析失败：file={self.corpus_path}, line={line_no}"
                        ) from exc

                    if not doc.doc_id:
                        raise BM25IndexError(
                            f"BM25 语料缺少 doc_id：file={self.corpus_path}, line={line_no}"
                        )

                    if doc.doc_id in doc_id_to_index:
                        logger.warning("检测到重复 doc_id，保留先出现者：%s", doc.doc_id)
                        continue

                    documents.append(doc)
                    doc_id_to_index[doc.doc_id] = len(documents) - 1
        except OSError as exc:
            raise BM25IndexError(f"读取 BM25 语料失败：{self.corpus_path}") from exc

        if not documents:
            raise BM25IndexError(f"BM25 语料为空：{self.corpus_path}")

        meta: dict[str, Any] = {}
        if self.meta_path.exists():
            try:
                meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
                if not isinstance(meta, dict):
                    meta = {}
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("读取 BM25 元信息失败，已忽略：%s, err=%s", self.meta_path, exc)
                meta = {}

        bm25_obj: BM25Okapi | None = None
        if self.index_path.exists():
            try:
                with self.index_path.open("rb") as f:
                    loaded = pickle.load(f)
                if isinstance(loaded, BM25Okapi):
                    bm25_obj = loaded
                else:
                    logger.warning("bm25_index.pkl 类型不符合预期，将自动重建。")
            except Exception as exc:
                logger.warning("加载 bm25_index.pkl 失败，将自动重建：err=%s", exc)

        if bm25_obj is None:
            tokenized_corpus = [doc.tokens for doc in documents]
            if not tokenized_corpus:
                raise BM25IndexError("BM25 tokenized corpus 为空，无法重建索引。")
            bm25_obj = BM25Okapi(tokenized_corpus)
            logger.info("已根据 bm25_corpus.jsonl 自动重建 BM25 索引。")

        self._documents = documents
        self._doc_id_to_index = doc_id_to_index
        self._bm25 = bm25_obj
        self._meta = meta

        logger.info(
            "BM25 索引加载完成：bm25_dir=%s, docs=%s",
            self.bm25_dir,
            len(self._documents),
        )

    def _ensure_loaded(self) -> None:
        """
        确保索引已加载。
        """
        if not self.is_loaded():
            self.load()

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[BM25SearchResult]:
        """
        执行 BM25 检索。

        参数：
            query: 用户原始问题
            top_k: 返回结果数量

        返回：
            list[BM25SearchResult]
        """
        if top_k <= 0:
            raise BM25IndexError("top_k 必须大于 0。")

        self._ensure_loaded()
        assert self._bm25 is not None  # 经过 _ensure_loaded 后必然存在

        raw_query = _normalize_text(query)
        if not raw_query:
            logger.warning("收到空 query，返回空结果。")
            return []

        normalized = normalize_query(raw_query)
        normalized_query = _normalize_text(normalized.get("normalized_query", ""))
        expanded_terms = _unique_keep_order(
            [str(item) for item in normalized.get("expanded_terms", [])]
        )
        guessed_categories = _unique_keep_order(
            [str(item) for item in normalized.get("guessed_categories", [])]
        )
        exact_terms = _unique_keep_order(
            [str(item) for item in normalized.get("exact_terms", [])]
        )
        special_exact_terms = _extract_special_exact_terms(raw_query)

        query_text = _build_query_text(normalized_query, expanded_terms)
        query_tokens = _tokenize_for_bm25(query_text)

        if not query_tokens:
            logger.warning("query 规范化后未产生有效 token，返回空结果。")
            return []

        try:
            raw_scores = self._bm25.get_scores(query_tokens)
        except Exception as exc:
            raise BM25IndexError("BM25 检索失败。") from exc

        results: list[BM25SearchResult] = []

        for idx, bm25_score in enumerate(raw_scores):
            score = _safe_float(bm25_score, 0.0)
            if score <= 0:
                continue

            doc = self._documents[idx]
            searchable_text = _make_searchable_text(doc)

            matched_general_terms = _match_general_terms(
                searchable_text,
                [normalized_query, *expanded_terms, *(guessed_categories or [])],
            )

            matched_exact_terms, exact_match_count = _count_exact_matches(searchable_text, exact_terms)
            matched_special_terms, special_match_count = _count_exact_matches(
                searchable_text,
                special_exact_terms,
            )

            exact_boost_value = (
                    exact_match_count * self.exact_term_boost
                    + special_match_count * self.special_exact_boost
            )

            category_boost_value = 0.0
            if guessed_categories and doc.category:
                if any(doc.category.casefold() == c.casefold() for c in guessed_categories):
                    category_boost_value = self.category_boost

            priority_boost_value = max(doc.priority - 1.0, 0.0) * self.priority_weight
            retrieval_score = score + exact_boost_value + category_boost_value + priority_boost_value

            results.append(
                BM25SearchResult(
                    doc_id=doc.doc_id,
                    chunk_hash=doc.chunk_hash,
                    retrieval_score=float(retrieval_score),
                    bm25_score=float(score),
                    exact_match_boost=float(exact_boost_value),
                    category_boost=float(category_boost_value),
                    priority_boost=float(priority_boost_value),
                    source_file=doc.source_file,
                    source_type=doc.source_type,
                    title=doc.title,
                    category=doc.category,
                    priority=float(doc.priority),
                    image_paths=doc.image_paths,
                    slide_no=doc.slide_no,
                    page_no=doc.page_no,
                    matched_terms=_unique_keep_order(matched_general_terms),
                    exact_terms=_unique_keep_order([*matched_exact_terms, *matched_special_terms]),
                    query_normalized=normalized_query,
                    snippet=doc.bm25_text,
                )
            )

        results.sort(
            key=lambda item: (
                item.retrieval_score,
                item.bm25_score,
                item.priority,
                -len(item.exact_terms),
            ),
            reverse=True,
        )

        final_results = results[:top_k]

        logger.debug(
            "BM25 检索完成：query=%s, normalized=%s, expanded=%s, exact=%s, results=%s",
            raw_query,
            normalized_query,
            expanded_terms,
            _unique_keep_order([*exact_terms, *special_exact_terms]),
            len(final_results),
        )
        return final_results