from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any

from app.clients.embedding_client import EmbeddingClient, EmbeddingClientError
from app.config import get_settings
from app.models import RetrievedChunk
from app.retrieval.bm25_index import BM25Index, BM25IndexError, BM25SearchResult
from app.retrieval.query_normalizer import QueryNormalizationResult, normalize_query
from app.retrieval.vector_store import VectorQueryResult, VectorStore, VectorStoreError

logger = logging.getLogger(__name__)

WHITESPACE_RE = re.compile(r"\s+")
IMAGE_INTENT_RE = re.compile(r"(截图|怎么操作|如何操作|看哪里|在哪看|哪里看|怎么点|点哪里|操作步骤)")
DEFAULT_IMAGE_INTENT_BOOST = 0.12
DEFAULT_RETURN_LIMIT_FALLBACK = 6
DEFAULT_VECTOR_TOP_K_FALLBACK = 8
DEFAULT_BM25_TOP_K_FALLBACK = 8


@dataclass
class _Candidate:
    """
    混合检索内部候选对象。
    用于融合向量召回与 BM25 召回结果，再映射为 RetrievedChunk。
    """

    doc_id: str
    chunk_hash: str = ""

    source_file: str = ""
    source_type: str = ""
    title: str = ""
    category: str | None = None
    full_text: str = ""
    image_paths: list[str] = field(default_factory=list)
    page_no: int | None = None
    slide_no: int | None = None
    priority: float = 1.0

    vector_distance: float | None = None
    vector_similarity: float = 0.0
    bm25_score: float = 0.0

    vector_hit: bool = False
    bm25_hit: bool = False

    matched_terms: list[str] = field(default_factory=list)
    exact_terms: list[str] = field(default_factory=list)

    base_score: float = 0.0
    retrieval_score: float = 0.0
    rerank_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class HybridRetriever:
    """
    企业网银 FAQ RAG demo 的混合检索器。

    检索流程：
    1. 规范化用户问题
    2. 用 normalized_query 生成 query embedding
    3. Chroma 向量召回
    4. BM25 稀疏召回
    5. 合并、去重、补齐元数据
    6. 执行规则重排
    7. 返回 RetrievedChunk 列表
    """

    def __init__(
            self,
            *,
            embedding_client: EmbeddingClient | None = None,
            vector_store: VectorStore | None = None,
            bm25_index: BM25Index | None = None,
            image_intent_boost: float = DEFAULT_IMAGE_INTENT_BOOST,
            vector_top_k: int | None = None,
            bm25_top_k: int | None = None,
            final_context_k: int | None = None,
            enable_debug: bool = True,
    ) -> None:
        settings = get_settings()

        self.embedding_client = embedding_client or EmbeddingClient()
        self.vector_store = vector_store or VectorStore()
        self.bm25_index = bm25_index or BM25Index()

        self.vector_top_k = int(
            vector_top_k
            or settings.retrieval.vector_top_k
            or DEFAULT_VECTOR_TOP_K_FALLBACK
        )
        self.bm25_top_k = int(
            bm25_top_k
            or settings.retrieval.bm25_top_k
            or DEFAULT_BM25_TOP_K_FALLBACK
        )
        self.final_context_k = int(
            final_context_k
            or settings.retrieval.final_context_k
            or DEFAULT_RETURN_LIMIT_FALLBACK
        )

        self.vector_weight = float(settings.rerank.vector_weight)
        self.bm25_weight = float(settings.rerank.bm25_weight)
        self.exact_match_boost = float(settings.rerank.exact_match_boost)
        self.category_boost = float(settings.rerank.category_boost)
        self.priority_boost = float(settings.rerank.priority_boost)
        self.image_intent_boost = float(image_intent_boost)

        self.source_priority = {
            "excel": float(settings.source_priority.excel),
            "ppt": float(settings.source_priority.ppt),
            "docx": float(settings.source_priority.docx),
        }

        self.enable_debug = enable_debug
        self.last_debug_info: dict[str, Any] = {}

    @staticmethod
    def _normalize_text(value: Any) -> str:
        """
        统一文本清洗。
        """
        if value is None:
            return ""
        text = str(value).replace("\u3000", " ")
        text = WHITESPACE_RE.sub(" ", text).strip()
        return text

    @staticmethod
    def _unique_keep_order(items: list[str]) -> list[str]:
        """
        保持原顺序去重。
        """
        result: list[str] = []
        seen: set[str] = set()

        for item in items:
            text = HybridRetriever._normalize_text(item)
            if not text:
                continue

            key = text.casefold()
            if key in seen:
                continue
            seen.add(key)
            result.append(text)

        return result

    @staticmethod
    def _coerce_str_list(value: Any) -> list[str]:
        """
        将 metadata 中可能出现的字符串或列表统一为字符串列表。
        """
        if value is None:
            return []

        if isinstance(value, list):
            return [
                item for item in
                (HybridRetriever._normalize_text(v) for v in value)
                if item
            ]

        text = HybridRetriever._normalize_text(value)
        return [text] if text else []

    @staticmethod
    def _safe_int(value: Any) -> int | None:
        """
        稳健转 int。
        """
        if value is None:
            return None
        text = HybridRetriever._normalize_text(value)
        if not text:
            return None
        try:
            return int(float(text))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        """
        稳健转 float。
        """
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _detect_image_intent(self, query: str) -> bool:
        """
        判断用户问题是否具有明显的“看图/步骤演示”意图。
        """
        return bool(IMAGE_INTENT_RE.search(self._normalize_text(query)))

    def _get_source_priority_value(self, source_type: str, priority: float) -> float:
        """
        获取用于重排的来源优先级。
        优先使用 chunk 自身 priority；若无效则回退到配置值。
        """
        if priority > 0:
            return priority
        return self.source_priority.get(source_type, 1.0)

    def _normalize_scores(self, values: list[float]) -> list[float]:
        """
        将一组分数缩放到 [0, 1]，便于向量分数和 BM25 分数融合。
        """
        if not values:
            return []

        valid_values = [float(v) for v in values]
        max_v = max(valid_values)
        min_v = min(valid_values)

        if math.isclose(max_v, min_v, rel_tol=1e-12, abs_tol=1e-12):
            return [1.0 if max_v > 0 else 0.0 for _ in valid_values]

        return [(v - min_v) / (max_v - min_v) for v in valid_values]

    def _build_candidate_from_vector(self, item: VectorQueryResult) -> _Candidate:
        """
        将向量召回结果转换为内部候选对象。
        """
        metadata = item.metadata or {}

        candidate = _Candidate(
            doc_id=self._normalize_text(metadata.get("doc_id")) or self._normalize_text(item.doc_id),
            chunk_hash=self._normalize_text(metadata.get("chunk_hash")),
            source_file=self._normalize_text(metadata.get("source_file")),
            source_type=self._normalize_text(metadata.get("source_type")).lower(),
            title=self._normalize_text(metadata.get("title")),
            category=self._normalize_text(metadata.get("category")) or None,
            full_text=self._normalize_text(item.document),
            image_paths=self._coerce_str_list(metadata.get("image_paths")),
            page_no=self._safe_int(metadata.get("page_no")),
            slide_no=self._safe_int(metadata.get("slide_no")),
            priority=self._safe_float(metadata.get("priority"), 1.0),
            vector_distance=item.distance,
            vector_similarity=self._safe_float(item.similarity, 0.0),
            vector_hit=True,
            metadata=dict(metadata),
        )
        return candidate

    def _build_candidate_from_bm25(self, item: BM25SearchResult) -> _Candidate:
        """
        将 BM25 召回结果转换为内部候选对象。
        """
        candidate = _Candidate(
            doc_id=self._normalize_text(item.doc_id),
            chunk_hash=self._normalize_text(item.chunk_hash),
            source_file=self._normalize_text(item.source_file),
            source_type=self._normalize_text(item.source_type).lower(),
            title=self._normalize_text(item.title),
            category=self._normalize_text(item.category) or None,
            full_text=self._normalize_text(item.snippet),
            image_paths=self._coerce_str_list(item.image_paths),
            page_no=item.page_no,
            slide_no=item.slide_no,
            priority=self._safe_float(item.priority, 1.0),
            bm25_score=self._safe_float(item.bm25_score, 0.0),
            bm25_hit=True,
            matched_terms=self._unique_keep_order(list(item.matched_terms)),
            exact_terms=self._unique_keep_order(list(item.exact_terms)),
        )
        return candidate

    def _merge_candidates(
            self,
            vector_results: list[VectorQueryResult],
            bm25_results: list[BM25SearchResult],
    ) -> dict[str, _Candidate]:
        """
        合并向量召回与 BM25 召回结果，并按 doc_id 去重。
        """
        merged: dict[str, _Candidate] = {}

        for item in vector_results:
            candidate = self._build_candidate_from_vector(item)
            if not candidate.doc_id:
                continue
            merged[candidate.doc_id] = candidate

        for item in bm25_results:
            candidate = self._build_candidate_from_bm25(item)
            if not candidate.doc_id:
                continue

            existing = merged.get(candidate.doc_id)
            if existing is None:
                merged[candidate.doc_id] = candidate
                continue

            existing.bm25_hit = True
            existing.bm25_score = max(existing.bm25_score, candidate.bm25_score)

            if not existing.source_file:
                existing.source_file = candidate.source_file
            if not existing.source_type:
                existing.source_type = candidate.source_type
            if not existing.title:
                existing.title = candidate.title
            if not existing.category:
                existing.category = candidate.category
            if not existing.full_text:
                existing.full_text = candidate.full_text
            if not existing.image_paths:
                existing.image_paths = candidate.image_paths
            if existing.page_no is None:
                existing.page_no = candidate.page_no
            if existing.slide_no is None:
                existing.slide_no = candidate.slide_no
            if existing.priority <= 0:
                existing.priority = candidate.priority
            if not existing.chunk_hash:
                existing.chunk_hash = candidate.chunk_hash

            existing.matched_terms = self._unique_keep_order(
                [*existing.matched_terms, *candidate.matched_terms]
            )
            existing.exact_terms = self._unique_keep_order(
                [*existing.exact_terms, *candidate.exact_terms]
            )

        return merged

    def _hydrate_missing_candidates(self, candidates: dict[str, _Candidate]) -> None:
        """
        对 BM25-only 候选补充 full_text 与 metadata。
        优先从 Chroma 按 doc_id 回查。
        """
        missing_ids: list[str] = []

        for doc_id, candidate in candidates.items():
            needs_hydrate = (
                    not candidate.full_text
                    or not candidate.title
                    or not candidate.source_file
                    or not candidate.source_type
            )
            if needs_hydrate:
                missing_ids.append(doc_id)

        if not missing_ids:
            return

        try:
            hydrated = self.vector_store.get_by_ids(missing_ids)
        except Exception as exc:
            logger.warning("回查 Chroma 补齐元数据失败：err=%s", exc)
            return

        for item in hydrated:
            candidate = candidates.get(item.doc_id)
            if candidate is None:
                continue

            metadata = item.metadata or {}
            if metadata:
                candidate.metadata.update(metadata)

            if not candidate.full_text:
                candidate.full_text = self._normalize_text(item.document)
            if not candidate.chunk_hash:
                candidate.chunk_hash = self._normalize_text(metadata.get("chunk_hash"))
            if not candidate.source_file:
                candidate.source_file = self._normalize_text(metadata.get("source_file"))
            if not candidate.source_type:
                candidate.source_type = self._normalize_text(metadata.get("source_type")).lower()
            if not candidate.title:
                candidate.title = self._normalize_text(metadata.get("title"))
            if not candidate.category:
                candidate.category = self._normalize_text(metadata.get("category")) or None
            if not candidate.image_paths:
                candidate.image_paths = self._coerce_str_list(metadata.get("image_paths"))
            if candidate.page_no is None:
                candidate.page_no = self._safe_int(metadata.get("page_no"))
            if candidate.slide_no is None:
                candidate.slide_no = self._safe_int(metadata.get("slide_no"))
            if candidate.priority <= 0:
                candidate.priority = self._safe_float(metadata.get("priority"), 1.0)

    def _build_searchable_text(self, candidate: _Candidate) -> str:
        """
        构造用于规则匹配的检索文本。
        """
        parts = [
            candidate.title,
            candidate.category or "",
            candidate.full_text,
            candidate.source_file,
            " ".join(candidate.image_paths),
            ]
        return " | ".join(part for part in parts if self._normalize_text(part)).casefold()

    def _rerank_candidates(
            self,
            candidates: dict[str, _Candidate],
            normalized: QueryNormalizationResult,
            raw_query: str,
    ) -> list[_Candidate]:
        """
        执行规则重排。
        """
        candidate_list = list(candidates.values())
        if not candidate_list:
            return []

        vector_norms = self._normalize_scores([c.vector_similarity for c in candidate_list])
        bm25_norms = self._normalize_scores([c.bm25_score for c in candidate_list])

        normalized_query = self._normalize_text(normalized.get("normalized_query", ""))
        guessed_categories = self._unique_keep_order(
            [str(item) for item in normalized.get("guessed_categories", [])]
        )
        exact_terms = self._unique_keep_order(
            [str(item) for item in normalized.get("exact_terms", [])]
        )
        image_intent = self._detect_image_intent(raw_query)

        for idx, candidate in enumerate(candidate_list):
            reasons: list[str] = []

            vector_part = vector_norms[idx] * self.vector_weight
            bm25_part = bm25_norms[idx] * self.bm25_weight
            final_score = vector_part + bm25_part

            if candidate.vector_hit:
                reasons.append(f"向量召回={vector_part:.4f}")
            if candidate.bm25_hit:
                reasons.append(f"BM25召回={bm25_part:.4f}")

            searchable_text = self._build_searchable_text(candidate)

            exact_hit_terms: list[str] = list(candidate.exact_terms)
            for term in exact_terms:
                clean_term = self._normalize_text(term)
                if clean_term and clean_term.casefold() in searchable_text:
                    exact_hit_terms.append(clean_term)
            exact_hit_terms = self._unique_keep_order(exact_hit_terms)

            if exact_hit_terms:
                boost = self.exact_match_boost * min(len(exact_hit_terms), 3)
                final_score += boost
                reasons.append(f"精确命中+{boost:.4f}({', '.join(exact_hit_terms[:3])})")
                candidate.exact_terms = exact_hit_terms

            category_hit_terms: list[str] = []
            if candidate.category and guessed_categories:
                for category in guessed_categories:
                    if candidate.category.casefold() == self._normalize_text(category).casefold():
                        category_hit_terms.append(category)

            if category_hit_terms:
                boost = self.category_boost
                final_score += boost
                reasons.append(f"分类命中+{boost:.4f}({', '.join(category_hit_terms)})")

            source_priority_value = self._get_source_priority_value(
                candidate.source_type,
                candidate.priority,
            )
            priority_delta = source_priority_value - 1.0
            if not math.isclose(priority_delta, 0.0, abs_tol=1e-12):
                boost = priority_delta * self.priority_boost
                final_score += boost
                reasons.append(f"来源优先级{boost:+.4f}")

            if image_intent and candidate.image_paths:
                boost = self.image_intent_boost
                final_score += boost
                reasons.append(f"图示意图+{boost:.4f}")

            if normalized_query and normalized_query.casefold() in searchable_text:
                phrase_boost = self.exact_match_boost * 0.5
                final_score += phrase_boost
                reasons.append(f"整句命中+{phrase_boost:.4f}")

            candidate.base_score = vector_part + bm25_part
            candidate.retrieval_score = final_score
            candidate.rerank_reason = "；".join(reasons) if reasons else "基础召回"

        candidate_list.sort(
            key=lambda c: (
                c.retrieval_score,
                c.base_score,
                c.priority,
                1 if c.image_paths else 0,
                c.title,
            ),
            reverse=True,
        )
        return candidate_list

    def _to_retrieved_chunk(self, candidate: _Candidate) -> RetrievedChunk:
        """
        将内部候选对象映射为 RetrievedChunk。
        """
        return RetrievedChunk(
            doc_id=candidate.doc_id,
            source_file=candidate.source_file,
            source_type=candidate.source_type,
            title=candidate.title,
            category=candidate.category,
            question=None,
            answer=None,
            full_text=candidate.full_text,
            keywords=[],
            image_paths=candidate.image_paths,
            page_no=candidate.page_no,
            slide_no=candidate.slide_no,
            priority=candidate.priority,
            chunk_hash=candidate.chunk_hash,
            retrieval_score=candidate.retrieval_score,
            vector_score=candidate.vector_similarity if candidate.vector_hit else None,
            bm25_score=candidate.bm25_score if candidate.bm25_hit else None,
            rerank_reason=candidate.rerank_reason,
        )

    def get_last_debug_info(self) -> dict[str, Any]:
        """
        返回最近一次 retrieve 的调试信息。
        """
        return dict(self.last_debug_info)

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        """
        执行混合检索并返回排序后的 RetrievedChunk 列表。
        """
        raw_query = self._normalize_text(query)
        debug_info: dict[str, Any] = {
            "raw_query": raw_query,
            "normalized": {},
            "vector_top_k": self.vector_top_k,
            "bm25_top_k": self.bm25_top_k,
            "final_context_k": self.final_context_k,
            "image_intent": self._detect_image_intent(raw_query),
            "errors": [],
            "counts": {},
        }

        if not raw_query:
            logger.warning("HybridRetriever 收到空 query，返回空结果。")
            self.last_debug_info = debug_info
            return []

        normalized = normalize_query(raw_query)
        debug_info["normalized"] = dict(normalized)

        normalized_query = self._normalize_text(normalized.get("normalized_query", ""))
        if not normalized_query:
            logger.warning("query 规范化后为空，返回空结果。")
            self.last_debug_info = debug_info
            return []

        vector_results: list[VectorQueryResult] = []
        bm25_results: list[BM25SearchResult] = []

        # 先做向量检索
        try:
            query_embedding = self.embedding_client.embed_text(normalized_query)
            if query_embedding:
                vector_results = self.vector_store.query_by_embedding(
                    query_embedding=query_embedding,
                    top_k=self.vector_top_k,
                    where=None,
                )
            else:
                debug_info["errors"].append("embedding_empty")
                logger.warning("query embedding 为空，跳过向量检索。")
        except (EmbeddingClientError, VectorStoreError) as exc:
            logger.warning("向量检索失败，将继续执行 BM25：err=%s", exc)
            debug_info["errors"].append(f"vector_error:{exc}")
        except Exception as exc:
            logger.exception("向量检索出现未预期异常：%s", exc)
            debug_info["errors"].append(f"vector_unexpected:{exc}")

        # 再做 BM25 检索
        try:
            bm25_results = self.bm25_index.search(
                query=raw_query,
                top_k=self.bm25_top_k,
            )
        except BM25IndexError as exc:
            logger.warning("BM25 检索失败：err=%s", exc)
            debug_info["errors"].append(f"bm25_error:{exc}")
        except Exception as exc:
            logger.exception("BM25 检索出现未预期异常：%s", exc)
            debug_info["errors"].append(f"bm25_unexpected:{exc}")

        merged = self._merge_candidates(
            vector_results=vector_results,
            bm25_results=bm25_results,
        )
        self._hydrate_missing_candidates(merged)

        reranked = self._rerank_candidates(
            candidates=merged,
            normalized=normalized,
            raw_query=raw_query,
        )

        final_chunks = [
            self._to_retrieved_chunk(candidate)
            for candidate in reranked[: self.final_context_k]
        ]

        debug_info["counts"] = {
            "vector_hits": len(vector_results),
            "bm25_hits": len(bm25_results),
            "merged_hits": len(merged),
            "final_hits": len(final_chunks),
        }

        if self.enable_debug:
            debug_info["top_results"] = [
                {
                    "doc_id": chunk.doc_id,
                    "title": chunk.title,
                    "source_type": chunk.source_type,
                    "category": chunk.category,
                    "retrieval_score": chunk.retrieval_score,
                    "vector_score": chunk.vector_score,
                    "bm25_score": chunk.bm25_score,
                    "rerank_reason": chunk.rerank_reason,
                }
                for chunk in final_chunks[:5]
            ]

        self.last_debug_info = debug_info

        logger.info(
            "混合检索完成：query=%s, vector_hits=%s, bm25_hits=%s, merged=%s, final=%s",
            raw_query,
            len(vector_results),
            len(bm25_results),
            len(merged),
            len(final_chunks),
        )
        return final_chunks