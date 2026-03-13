from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection
from pydantic import BaseModel, Field

from app.config import get_settings

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION_NAME = "ebank_faq_kb"
DEFAULT_TOP_K = 8


class VectorStoreError(RuntimeError):
    """向量库访问异常。"""


class VectorQueryResult(BaseModel):
    """
    单条向量检索结果。

    字段说明：
    - doc_id: 文档主键，通常对应 KBChunk.doc_id
    - distance: Chroma 返回的距离值，越小通常越相近
    - similarity: 基于 distance 计算的简单相似度，范围不固定，但数值越大代表越相近
    - document: 原始文本内容
    - metadata: Chroma 中保存的元数据
    """

    doc_id: str
    distance: float = 0.0
    similarity: float = 0.0
    document: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class VectorStore:
    """
    本地 Chroma 向量库访问层。

    功能：
    1. 连接本地 data/index/chroma_db
    2. 获取指定 collection
    3. 支持按 embedding 检索
    4. 支持按 ids 读取文档与 metadata
    """

    def __init__(
            self,
            chroma_dir: Path | None = None,
            *,
            collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> None:
        settings = get_settings()

        self.chroma_dir = (chroma_dir or settings.paths.chroma_db_dir).expanduser().resolve()
        self.collection_name = collection_name.strip() or DEFAULT_COLLECTION_NAME

        if not self.chroma_dir.exists():
            raise VectorStoreError(
                f"Chroma 数据库目录不存在：{self.chroma_dir}。"
                "请先执行离线索引构建。"
            )

        if not self.chroma_dir.is_dir():
            raise VectorStoreError(
                f"Chroma 路径不是目录：{self.chroma_dir}"
            )

        try:
            self._client = chromadb.PersistentClient(path=str(self.chroma_dir))
        except Exception as exc:
            raise VectorStoreError(
                f"初始化 Chroma PersistentClient 失败：{self.chroma_dir}"
            ) from exc

        self._collection = self._load_collection(self.collection_name)

        logger.info(
            "VectorStore 初始化完成：chroma_dir=%s, collection=%s",
            self.chroma_dir,
            self.collection_name,
        )

    @property
    def collection(self) -> Collection:
        """
        当前使用的 Chroma collection。
        """
        return self._collection

    def _load_collection(self, collection_name: str) -> Collection:
        """
        加载指定 collection。
        若 collection 不存在，抛出明确异常。
        """
        try:
            collection = self._client.get_collection(name=collection_name)
        except ValueError as exc:
            raise VectorStoreError(
                f"Chroma collection 不存在：{collection_name}。"
                "请确认已执行 build_indexes.py，且 collection 名称一致。"
            ) from exc
        except Exception as exc:
            raise VectorStoreError(
                f"加载 Chroma collection 失败：{collection_name}"
            ) from exc

        return collection

    @staticmethod
    def _normalize_embedding(query_embedding: list[float]) -> list[float]:
        """
        校验并规范化查询向量。
        """
        if not isinstance(query_embedding, list):
            raise VectorStoreError("query_embedding 必须是 list[float]。")

        if not query_embedding:
            raise VectorStoreError("query_embedding 不能为空。")

        normalized: list[float] = []
        for idx, value in enumerate(query_embedding):
            try:
                normalized.append(float(value))
            except (TypeError, ValueError) as exc:
                raise VectorStoreError(
                    f"query_embedding 第 {idx} 个元素无法转换为 float。"
                ) from exc

        return normalized

    @staticmethod
    def _normalize_ids(ids: list[str]) -> list[str]:
        """
        校验并清洗 id 列表。
        """
        if not isinstance(ids, list):
            raise VectorStoreError("ids 必须是 list[str]。")

        cleaned: list[str] = []
        for item in ids:
            text = str(item).strip()
            if text:
                cleaned.append(text)

        if not cleaned:
            raise VectorStoreError("ids 不能为空。")

        # 保持顺序去重
        result: list[str] = []
        seen: set[str] = set()
        for item in cleaned:
            if item in seen:
                continue
            seen.add(item)
            result.append(item)

        return result

    @staticmethod
    def _normalize_where(where: dict[str, Any] | None) -> dict[str, Any] | None:
        """
        校验 metadata 过滤条件。
        """
        if where is None:
            return None

        if not isinstance(where, dict):
            raise VectorStoreError("where 必须是 dict 或 None。")

        if not where:
            return None

        return where

    @staticmethod
    def _to_python_list(value: Any) -> list[Any]:
        """
        将 Chroma 可能返回的 list / tuple / numpy 数组等结构统一转为 Python list。
        """
        if value is None:
            return []

        if isinstance(value, list):
            return value

        if isinstance(value, tuple):
            return list(value)

        tolist = getattr(value, "tolist", None)
        if callable(tolist):
            converted = tolist()
            if isinstance(converted, list):
                return converted
            if isinstance(converted, tuple):
                return list(converted)
            return [converted]

        try:
            return list(value)
        except TypeError:
            return [value]

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        """
        稳健地将值转为 float。
        """
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _distance_to_similarity(distance: float) -> float:
        """
        将距离值映射为一个单调递减的相似度值。

        说明：
        - 这里只做轻量统一映射，便于后续 hybrid merge
        - 不假设具体距离度量类型，仅保证“距离越小，相似度越大”
        """
        if distance < 0:
            return 0.0
        return 1.0 / (1.0 + distance)

    def query_by_embedding(
            self,
            query_embedding: list[float],
            top_k: int = DEFAULT_TOP_K,
            where: dict[str, Any] | None = None,
    ) -> list[VectorQueryResult]:
        """
        按查询向量执行相似度检索。

        参数：
        - query_embedding: 单条查询向量
        - top_k: 返回结果数量
        - where: Chroma metadata 过滤条件，例如 {"source_type": "ppt"}

        返回：
        - list[VectorQueryResult]
        """
        if top_k <= 0:
            raise VectorStoreError("top_k 必须大于 0。")

        embedding = self._normalize_embedding(query_embedding)
        where_filter = self._normalize_where(where)

        logger.debug(
            "开始执行向量检索：collection=%s, top_k=%s, where=%s",
            self.collection_name,
            top_k,
            where_filter,
        )

        try:
            result = self._collection.query(
                query_embeddings=[embedding],
                n_results=top_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise VectorStoreError("Chroma query 执行失败。") from exc

        ids_nested = self._to_python_list(result.get("ids"))
        docs_nested = self._to_python_list(result.get("documents"))
        metas_nested = self._to_python_list(result.get("metadatas"))
        dists_nested = self._to_python_list(result.get("distances"))

        ids = self._to_python_list(ids_nested[0]) if ids_nested else []
        docs = self._to_python_list(docs_nested[0]) if docs_nested else []
        metas = self._to_python_list(metas_nested[0]) if metas_nested else []
        dists = self._to_python_list(dists_nested[0]) if dists_nested else []

        size = len(ids)
        if not (len(docs) == len(metas) == len(dists) == size):
            raise VectorStoreError(
                "Chroma query 返回结构异常：ids/documents/metadatas/distances 长度不一致。"
            )

        results: list[VectorQueryResult] = []
        for idx in range(size):
            doc_id = str(ids[idx]).strip()
            document = "" if docs[idx] is None else str(docs[idx])
            metadata = metas[idx] if isinstance(metas[idx], dict) else {}
            distance = self._safe_float(dists[idx], 0.0)
            similarity = self._distance_to_similarity(distance)

            results.append(
                VectorQueryResult(
                    doc_id=doc_id,
                    distance=distance,
                    similarity=similarity,
                    document=document,
                    metadata=metadata,
                )
            )

        logger.debug(
            "向量检索完成：collection=%s, returned=%s",
            self.collection_name,
            len(results),
        )
        return results

    def get_by_ids(self, ids: list[str]) -> list[VectorQueryResult]:
        """
        根据 doc_id 列表批量读取文档与 metadata。

        参数：
        - ids: doc_id 列表

        返回：
        - list[VectorQueryResult]
          其中 distance / similarity 置为 0.0，因为这是直接读取而非相似度查询
        """
        normalized_ids = self._normalize_ids(ids)

        logger.debug(
            "开始按 ids 获取文档：collection=%s, count=%s",
            self.collection_name,
            len(normalized_ids),
        )

        try:
            result = self._collection.get(
                ids=normalized_ids,
                include=["documents", "metadatas"],
            )
        except Exception as exc:
            raise VectorStoreError("Chroma get 执行失败。") from exc

        result_ids = self._to_python_list(result.get("ids"))
        result_docs = self._to_python_list(result.get("documents"))
        result_metas = self._to_python_list(result.get("metadatas"))

        size = len(result_ids)
        if not (len(result_docs) == len(result_metas) == size):
            raise VectorStoreError(
                "Chroma get 返回结构异常：ids/documents/metadatas 长度不一致。"
            )

        results: list[VectorQueryResult] = []
        for idx in range(size):
            doc_id = str(result_ids[idx]).strip()
            document = "" if result_docs[idx] is None else str(result_docs[idx])
            metadata = result_metas[idx] if isinstance(result_metas[idx], dict) else {}

            results.append(
                VectorQueryResult(
                    doc_id=doc_id,
                    distance=0.0,
                    similarity=0.0,
                    document=document,
                    metadata=metadata,
                )
            )

        logger.debug(
            "按 ids 获取完成：collection=%s, returned=%s",
            self.collection_name,
            len(results),
        )
        return results