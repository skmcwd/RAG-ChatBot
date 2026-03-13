from __future__ import annotations

import argparse
import json
import logging
import pickle
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection
from pydantic import BaseModel, Field, ValidationError
from rank_bm25 import BM25Okapi

# 兼容直接执行：
# python scripts/build_indexes.py --input data/parsed/kb.jsonl
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.clients.embedding_client import EmbeddingClient, EmbeddingClientError  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.logging_utils import setup_logging  # noqa: E402
from app.models import KBChunk  # noqa: E402

logger = logging.getLogger(__name__)

WHITESPACE_RE = re.compile(r"\s+")
PUNCT_SPLIT_RE = re.compile(r"[，,。；;：:、/\\|（）()\[\]【】《》<>\-—_~!！?？\"'“”‘’\s]+")
ASCII_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
CJK_SPAN_RE = re.compile(r"[\u4e00-\u9fff]+")

DEFAULT_COLLECTION_NAME = "ebank_faq_kb"
DEFAULT_BATCH_SIZE = 64
DEFAULT_PRIORITY_BY_SOURCE_TYPE: dict[str, float] = {
    "docx": 1.1,
    "excel": 1.0,
    "ppt": 0.95,
}


class IndexBuildError(RuntimeError):
    """索引构建异常。"""


class ManifestRecord(BaseModel):
    """
    索引清单中的单条记录。
    用于支持增量构建与重复运行时的去重。
    """

    doc_id: str
    chunk_hash: str
    source_file: str = ""
    source_type: str = ""
    title: str = ""
    priority: float = 1.0
    indexed_at: str = ""


class IndexManifest(BaseModel):
    """
    本地索引清单。
    记录哪些 chunk 已经写入索引，便于增量构建。
    """

    version: int = 1
    collection_name: str = DEFAULT_COLLECTION_NAME
    records: list[ManifestRecord] = Field(default_factory=list)
    updated_at: str = ""


class BM25Entry(BaseModel):
    """
    BM25 检索侧存储的轻量记录。
    """

    doc_id: str
    chunk_hash: str
    bm25_text: str
    tokens: list[str] = Field(default_factory=list)
    source_file: str = ""
    source_type: str = ""
    title: str = ""
    category: str | None = None
    priority: float = 1.0
    image_paths: list[str] = Field(default_factory=list)
    slide_no: int | None = None
    page_no: int | None = None


class BuildStats(BaseModel):
    """
    构建统计信息。
    """

    input_total: int = 0
    input_valid: int = 0
    input_invalid: int = 0
    input_dedup_skipped: int = 0

    manifest_existing_skipped: int = 0
    chroma_existing_id_skipped: int = 0
    chroma_new_candidates: int = 0

    embedding_success: int = 0
    embedding_failed: int = 0
    chroma_upserted: int = 0
    chroma_failed: int = 0

    bm25_total_docs: int = 0
    bm25_new_docs: int = 0

    final_manifest_records: int = 0
    by_source_type: dict[str, int] = Field(default_factory=dict)
    by_category: dict[str, int] = Field(default_factory=dict)


def normalize_text(value: Any) -> str:
    """
    统一文本清洗：
    1. None 安全处理
    2. 全角空格转半角空格
    3. 连续空白折叠为单空格
    4. 去掉首尾空白
    """
    if value is None:
        return ""
    text = str(value).replace("\u3000", " ")
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def unique_keep_order(items: list[str]) -> list[str]:
    """
    保持原顺序去重。
    """
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def ensure_dir(path: Path) -> None:
    """
    确保目录存在。
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise IndexBuildError(f"创建目录失败：{path}") from exc


def now_iso() -> str:
    """
    返回 UTC ISO 时间字符串。
    """
    return datetime.now(timezone.utc).isoformat()


def stable_chunk_hash(source_file: str, title: str, full_text: str) -> str:
    """
    生成稳定哈希。
    """
    import hashlib

    payload = "||".join(
        [
            normalize_text(source_file).casefold(),
            normalize_text(title),
            normalize_text(full_text),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def coerce_str_list(value: Any) -> list[str]:
    """
    将输入转换为字符串列表。
    """
    if value is None:
        return []

    if isinstance(value, list):
        return [item for item in (normalize_text(v) for v in value) if item]

    text = normalize_text(value)
    return [text] if text else []


def default_priority_for_source_type(source_type: str) -> float:
    """
    获取来源类型默认优先级。
    """
    return DEFAULT_PRIORITY_BY_SOURCE_TYPE.get(source_type, 1.0)


def derive_keywords(chunk: KBChunk) -> list[str]:
    """
    当 keywords 缺失时，用轻量规则自动补全。
    """
    candidates = [
        normalize_text(chunk.category),
        normalize_text(chunk.title),
        normalize_text(chunk.question),
        normalize_text(chunk.answer),
        normalize_text(chunk.source_type),
    ]

    parts: list[str] = []
    for text in candidates:
        if not text:
            continue
        parts.extend(PUNCT_SPLIT_RE.split(text))
        parts.extend(ASCII_TOKEN_RE.findall(text))

    result: list[str] = []
    seen: set[str] = set()
    for part in parts:
        token = normalize_text(part)
        if not token:
            continue
        if len(token) < 2 or len(token) > 24:
            continue
        key = token.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(token)
        if len(result) >= 12:
            break

    return result


def canonicalize_chunk(raw: dict[str, Any]) -> KBChunk:
    """
    将原始 JSON 记录校验并标准化为 KBChunk。
    """
    try:
        base_chunk = KBChunk.model_validate(raw)
    except ValidationError as exc:
        raise IndexBuildError("KBChunk 校验失败。") from exc

    source_file = normalize_text(base_chunk.source_file)
    source_type = normalize_text(base_chunk.source_type).lower() or "excel"
    title = normalize_text(base_chunk.title)
    full_text = normalize_text(base_chunk.full_text)

    if not source_file:
        raise IndexBuildError("KBChunk 缺少 source_file。")
    if not title:
        raise IndexBuildError(f"KBChunk 缺少 title：source_file={source_file}")
    if not full_text:
        raise IndexBuildError(f"KBChunk 缺少 full_text：source_file={source_file}, title={title}")

    chunk_hash = normalize_text(base_chunk.chunk_hash) or stable_chunk_hash(
        source_file=source_file,
        title=title,
        full_text=full_text,
    )

    priority = (
        float(base_chunk.priority)
        if base_chunk.priority and float(base_chunk.priority) > 0
        else default_priority_for_source_type(source_type)
    )

    doc_id = normalize_text(base_chunk.doc_id)
    if not doc_id:
        doc_id = f"{Path(source_file).stem}-{chunk_hash[:12]}"

    image_paths = unique_keep_order(coerce_str_list(base_chunk.image_paths))
    keywords = unique_keep_order(base_chunk.keywords or []) or derive_keywords(base_chunk)

    return KBChunk(
        doc_id=doc_id,
        source_file=source_file,
        source_type=source_type,
        title=title,
        category=normalize_text(base_chunk.category) or None,
        question=normalize_text(base_chunk.question) or None,
        answer=normalize_text(base_chunk.answer) or None,
        full_text=full_text,
        keywords=keywords,
        image_paths=image_paths,
        page_no=base_chunk.page_no,
        slide_no=base_chunk.slide_no,
        priority=priority,
        chunk_hash=chunk_hash,
    )


def load_kb_chunks(input_path: Path) -> tuple[list[KBChunk], BuildStats]:
    """
    从 kb.jsonl 加载知识块，并做基础校验。
    """
    if not input_path.exists():
        raise IndexBuildError(f"输入文件不存在：{input_path}")
    if not input_path.is_file():
        raise IndexBuildError(f"输入路径不是文件：{input_path}")

    stats = BuildStats()
    chunks: list[KBChunk] = []

    try:
        with input_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                text = line.strip()
                if not text:
                    continue

                stats.input_total += 1

                try:
                    obj = json.loads(text)
                    if not isinstance(obj, dict):
                        raise IndexBuildError("JSONL 行不是对象。")
                    chunk = canonicalize_chunk(obj)
                except Exception as exc:
                    stats.input_invalid += 1
                    logger.warning("跳过无效记录：line=%s, err=%s", line_no, exc)
                    continue

                chunks.append(chunk)
                stats.input_valid += 1
    except OSError as exc:
        raise IndexBuildError(f"读取 kb.jsonl 失败：{input_path}") from exc

    return chunks, stats


def deduplicate_chunks(chunks: list[KBChunk], stats: BuildStats) -> list[KBChunk]:
    """
    按 doc_id 与 chunk_hash 去重。
    若发生冲突，保留 priority 更高者；priority 相同则保留先出现者。
    """
    selected_by_doc_id: dict[str, KBChunk] = {}
    selected_by_hash: dict[str, KBChunk] = {}

    for chunk in chunks:
        existing_by_doc = selected_by_doc_id.get(chunk.doc_id)
        if existing_by_doc is not None:
            stats.input_dedup_skipped += 1
            if chunk.priority > existing_by_doc.priority:
                selected_by_doc_id[chunk.doc_id] = chunk
            continue

        existing_by_hash = selected_by_hash.get(chunk.chunk_hash)
        if existing_by_hash is not None:
            stats.input_dedup_skipped += 1
            if chunk.priority > existing_by_hash.priority:
                # 替换 hash 命中时，需要同步清掉旧 doc_id
                old_doc_id = existing_by_hash.doc_id
                selected_by_doc_id.pop(old_doc_id, None)
                selected_by_doc_id[chunk.doc_id] = chunk
                selected_by_hash[chunk.chunk_hash] = chunk
            continue

        selected_by_doc_id[chunk.doc_id] = chunk
        selected_by_hash[chunk.chunk_hash] = chunk

    return list(selected_by_doc_id.values())


def load_manifest(manifest_path: Path, collection_name: str) -> IndexManifest:
    """
    读取本地索引清单。
    """
    if not manifest_path.exists():
        return IndexManifest(collection_name=collection_name, updated_at=now_iso())

    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = IndexManifest.model_validate(data)
    except Exception as exc:
        raise IndexBuildError(f"读取索引清单失败：{manifest_path}") from exc

    if manifest.collection_name != collection_name:
        logger.warning(
            "检测到 manifest 的 collection_name=%s 与当前=%s 不一致，仍继续使用当前 collection_name。",
            manifest.collection_name,
            collection_name,
        )
        manifest.collection_name = collection_name

    return manifest


def save_manifest(manifest: IndexManifest, manifest_path: Path) -> None:
    """
    保存索引清单。
    """
    ensure_dir(manifest_path.parent)
    manifest.updated_at = now_iso()

    try:
        manifest_path.write_text(
            json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        raise IndexBuildError(f"写入索引清单失败：{manifest_path}") from exc


def manifest_doc_ids(manifest: IndexManifest) -> set[str]:
    """
    提取 manifest 中已有 doc_id 集合。
    """
    return {item.doc_id for item in manifest.records}


def manifest_chunk_hashes(manifest: IndexManifest) -> set[str]:
    """
    提取 manifest 中已有 chunk_hash 集合。
    """
    return {item.chunk_hash for item in manifest.records}


def append_manifest_records(manifest: IndexManifest, chunks: list[KBChunk]) -> None:
    """
    将新写入索引的 chunk 追加到 manifest。
    """
    existing_ids = manifest_doc_ids(manifest)
    existing_hashes = manifest_chunk_hashes(manifest)

    for chunk in chunks:
        if chunk.doc_id in existing_ids or chunk.chunk_hash in existing_hashes:
            continue

        manifest.records.append(
            ManifestRecord(
                doc_id=chunk.doc_id,
                chunk_hash=chunk.chunk_hash,
                source_file=chunk.source_file,
                source_type=chunk.source_type,
                title=chunk.title,
                priority=chunk.priority,
                indexed_at=now_iso(),
            )
        )
        existing_ids.add(chunk.doc_id)
        existing_hashes.add(chunk.chunk_hash)


def load_existing_bm25_entries(corpus_path: Path) -> dict[str, BM25Entry]:
    """
    读取已有 BM25 语料条目。
    键为 doc_id。
    """
    if not corpus_path.exists():
        return {}

    entries: dict[str, BM25Entry] = {}
    try:
        with corpus_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    obj = json.loads(text)
                    entry = BM25Entry.model_validate(obj)
                    entries[entry.doc_id] = entry
                except Exception as exc:
                    logger.warning("忽略损坏的 BM25 语料记录：line=%s, err=%s", line_no, exc)
    except OSError as exc:
        raise IndexBuildError(f"读取 BM25 语料文件失败：{corpus_path}") from exc

    return entries


def build_bm25_text(chunk: KBChunk) -> str:
    """
    为 BM25 构造稀疏检索文本。
    按要求优先拼接：
    title + question + answer + keywords + category
    """
    parts: list[str] = [chunk.title]

    if chunk.question:
        parts.append(chunk.question)
    if chunk.answer:
        parts.append(chunk.answer)
    if chunk.keywords:
        parts.append(" ".join(chunk.keywords))
    if chunk.category:
        parts.append(chunk.category)

    # 对于没有 question/answer 的手册型 chunk，用 full_text 补足检索语义
    if not chunk.question and not chunk.answer:
        parts.append(chunk.full_text)

    text = "。".join(part.strip("。") for part in parts if normalize_text(part)).strip("。")
    return f"{text}。" if text else ""


def tokenize_for_bm25(text: str) -> list[str]:
    """
    轻量中英混合分词。
    不依赖第三方分词库，保证单机演示版稳定运行。

    规则：
    1. 提取英文/数字 token
    2. 对中文连续片段，加入整段（长度较短时）以及 2-gram / 3-gram
    3. 结果去重但保留局部顺序
    """
    normalized = normalize_text(text).lower()
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

        # 2-gram
        if len(span) >= 2:
            tokens.extend(span[i: i + 2] for i in range(len(span) - 1))

        # 3-gram
        if len(span) >= 3:
            tokens.extend(span[i: i + 3] for i in range(len(span) - 2))

    # 对被标点切开的普通片段也保留
    for part in PUNCT_SPLIT_RE.split(normalized):
        part = normalize_text(part)
        if 2 <= len(part) <= 24:
            tokens.append(part)

    filtered: list[str] = []
    for token in tokens:
        token = normalize_text(token)
        if not token:
            continue
        filtered.append(token)

    return filtered


def build_bm25_entry(chunk: KBChunk) -> BM25Entry:
    """
    从 KBChunk 构造 BM25Entry。
    """
    bm25_text = build_bm25_text(chunk)
    tokens = tokenize_for_bm25(bm25_text)

    return BM25Entry(
        doc_id=chunk.doc_id,
        chunk_hash=chunk.chunk_hash,
        bm25_text=bm25_text,
        tokens=tokens,
        source_file=chunk.source_file,
        source_type=chunk.source_type,
        title=chunk.title,
        category=chunk.category,
        priority=chunk.priority,
        image_paths=chunk.image_paths,
        slide_no=chunk.slide_no,
        page_no=chunk.page_no,
    )


def save_bm25_artifacts(
        entries: list[BM25Entry],
        bm25_dir: Path,
) -> None:
    """
    保存 BM25 检索所需文件：
    1. bm25_corpus.jsonl：语料与 token
    2. bm25_index.pkl：BM25Okapi 对象
    3. bm25_meta.json：简单元信息
    """
    ensure_dir(bm25_dir)

    corpus_path = bm25_dir / "bm25_corpus.jsonl"
    index_path = bm25_dir / "bm25_index.pkl"
    meta_path = bm25_dir / "bm25_meta.json"

    tokenized_corpus = [entry.tokens for entry in entries]
    bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else BM25Okapi([["空"]])

    try:
        with corpus_path.open("w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry.model_dump(mode="json"), ensure_ascii=False) + "\n")
    except OSError as exc:
        raise IndexBuildError(f"写入 BM25 语料失败：{corpus_path}") from exc

    try:
        with index_path.open("wb") as f:
            pickle.dump(bm25, f)
    except OSError as exc:
        raise IndexBuildError(f"写入 BM25 索引失败：{index_path}") from exc

    meta = {
        "updated_at": now_iso(),
        "doc_count": len(entries),
        "has_real_docs": bool(entries),
    }
    try:
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:
        raise IndexBuildError(f"写入 BM25 元信息失败：{meta_path}") from exc


def make_chroma_metadata(chunk: KBChunk) -> dict[str, Any]:
    """
    构造写入 Chroma 的 metadata。
    仅包含扁平可序列化字段。
    """
    metadata: dict[str, Any] = {
        "doc_id": chunk.doc_id,
        "chunk_hash": chunk.chunk_hash,
        "source_file": chunk.source_file,
        "source_type": chunk.source_type,
        "title": chunk.title,
        "priority": float(chunk.priority),
        "image_paths": chunk.image_paths,
        "keywords": chunk.keywords,
    }

    if chunk.category is not None:
        metadata["category"] = chunk.category
    if chunk.slide_no is not None:
        metadata["slide_no"] = int(chunk.slide_no)
    if chunk.page_no is not None:
        metadata["page_no"] = int(chunk.page_no)

    return metadata


def get_chroma_client(chroma_dir: Path) -> chromadb.PersistentClient:
    """
    创建本地 Chroma PersistentClient。
    """
    ensure_dir(chroma_dir)
    try:
        client = chromadb.PersistentClient(path=chroma_dir)
    except Exception as exc:
        raise IndexBuildError(f"初始化 Chroma PersistentClient 失败：{chroma_dir}") from exc
    return client


def get_or_reset_collection(
        client: chromadb.PersistentClient,
        collection_name: str,
        *,
        rebuild: bool,
) -> Collection:
    """
    获取或重建 Chroma collection。
    """
    if rebuild:
        try:
            client.delete_collection(collection_name)
            logger.info("已删除旧 collection：%s", collection_name)
        except Exception:
            logger.info("未发现旧 collection，直接创建新 collection：%s", collection_name)

    try:
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={
                "description": "Enterprise banking FAQ RAG demo knowledge base",
            },
        )
    except Exception as exc:
        raise IndexBuildError(f"获取或创建 Chroma collection 失败：{collection_name}") from exc

    return collection


def get_chroma_max_batch_size(client: chromadb.PersistentClient, fallback: int) -> int:
    """
    获取 Chroma 允许的最大 batch size。
    """
    try:
        value = client.get_max_batch_size()
        if isinstance(value, int) and value > 0:
            return value
    except Exception:
        pass
    return fallback


def chunked(items: list[KBChunk], batch_size: int) -> list[list[KBChunk]]:
    """
    简单分批。
    """
    return [items[i: i + batch_size] for i in range(0, len(items), batch_size)]


def filter_new_chunks_by_manifest(
        chunks: list[KBChunk],
        manifest: IndexManifest,
        stats: BuildStats,
) -> list[KBChunk]:
    """
    基于 manifest 先做一轮增量过滤。
    """
    existing_doc_ids = manifest_doc_ids(manifest)
    existing_hashes = manifest_chunk_hashes(manifest)

    new_chunks: list[KBChunk] = []
    for chunk in chunks:
        if chunk.doc_id in existing_doc_ids or chunk.chunk_hash in existing_hashes:
            stats.manifest_existing_skipped += 1
            continue
        new_chunks.append(chunk)

    return new_chunks


def filter_existing_doc_ids_in_chroma(
        collection: Collection,
        chunks: list[KBChunk],
        query_batch_size: int,
        stats: BuildStats,
) -> list[KBChunk]:
    """
    再基于 Chroma 当前 collection 中已存在的 doc_id 做过滤。
    用于应对 manifest 与实际索引不完全一致的情况。
    """
    if not chunks:
        return []

    result: list[KBChunk] = []

    for batch in chunked(chunks, query_batch_size):
        ids = [chunk.doc_id for chunk in batch]
        existing_ids: set[str] = set()

        try:
            data = collection.get(ids=ids, include=["metadatas"])
            existing_ids = set(data.get("ids", []) or [])
        except Exception as exc:
            logger.warning("查询 Chroma 现有 doc_id 失败，退化为仅依赖 manifest：err=%s", exc)

        for chunk in batch:
            if chunk.doc_id in existing_ids:
                stats.chroma_existing_id_skipped += 1
                continue
            result.append(chunk)

    return result


def upsert_chunks_to_chroma(
        collection: Collection,
        chunks: list[KBChunk],
        embedder: EmbeddingClient,
        batch_size: int,
        stats: BuildStats,
) -> list[KBChunk]:
    """
    将新增 chunk 写入 Chroma。
    返回实际写入成功的 chunk 列表。
    """
    if not chunks:
        return []

    inserted: list[KBChunk] = []

    for batch_idx, batch in enumerate(chunked(chunks, batch_size), start=1):
        texts = [chunk.full_text for chunk in batch]

        try:
            embeddings = embedder.embed_texts(texts)
        except EmbeddingClientError as exc:
            stats.chroma_failed += len(batch)
            stats.embedding_failed += len(batch)
            logger.error("Embedding 批处理失败，整批跳过：batch=%s, err=%s", batch_idx, exc)
            continue
        except Exception as exc:
            stats.chroma_failed += len(batch)
            stats.embedding_failed += len(batch)
            logger.exception("Embedding 批处理出现异常，整批跳过：batch=%s, err=%s", batch_idx, exc)
            continue

        valid_ids: list[str] = []
        valid_documents: list[str] = []
        valid_metadatas: list[dict[str, Any]] = []
        valid_embeddings: list[list[float]] = []
        valid_chunks: list[KBChunk] = []

        for chunk, vector in zip(batch, embeddings, strict=True):
            if not vector:
                stats.embedding_failed += 1
                logger.warning("收到空 embedding，跳过：doc_id=%s", chunk.doc_id)
                continue

            stats.embedding_success += 1
            valid_ids.append(chunk.doc_id)
            valid_documents.append(chunk.full_text)
            valid_metadatas.append(make_chroma_metadata(chunk))
            valid_embeddings.append(vector)
            valid_chunks.append(chunk)

        if not valid_ids:
            continue

        try:
            collection.upsert(
                ids=valid_ids,
                documents=valid_documents,
                metadatas=valid_metadatas,
                embeddings=valid_embeddings,
            )
        except Exception as exc:
            stats.chroma_failed += len(valid_ids)
            logger.exception("Chroma upsert 失败：batch=%s, err=%s", batch_idx, exc)
            continue

        stats.chroma_upserted += len(valid_ids)
        inserted.extend(valid_chunks)

        logger.info(
            "Chroma 写入完成：batch=%s, inserted=%s",
            batch_idx,
            len(valid_ids),
        )

    return inserted


def write_stats(stats: BuildStats, output_path: Path) -> None:
    """
    写出构建统计文件。
    """
    ensure_dir(output_path.parent)
    try:
        output_path.write_text(
            json.dumps(stats.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        raise IndexBuildError(f"写入统计文件失败：{output_path}") from exc


def parse_args() -> argparse.Namespace:
    """
    命令行参数解析。
    """
    parser = argparse.ArgumentParser(
        description="离线构建企业网银 FAQ RAG demo 的 Chroma 向量库和 BM25 索引。"
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="输入 kb.jsonl 文件路径，例如 data/parsed/kb.jsonl",
    )
    parser.add_argument(
        "--collection-name",
        default=DEFAULT_COLLECTION_NAME,
        help=f"Chroma collection 名称，默认 {DEFAULT_COLLECTION_NAME}",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Embedding / Chroma 写入批大小，默认 {DEFAULT_BATCH_SIZE}",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="是否重建索引。开启后会删除旧的 Chroma collection，并重建 BM25 与 manifest。",
    )
    parser.add_argument(
        "--stats-output",
        type=Path,
        default=None,
        help="统计文件输出路径，默认 data/index/build_stats.json",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="日志级别，例如 DEBUG / INFO / WARNING / ERROR",
    )
    return parser.parse_args()


def main() -> int:
    """
    命令行主入口。
    """
    args = parse_args()
    setup_logging(args.log_level)

    if args.batch_size <= 0:
        logger.error("--batch-size 必须大于 0。")
        return 1

    settings = get_settings()

    input_path = args.input.expanduser().resolve()
    chroma_dir = settings.paths.chroma_db_dir
    bm25_dir = settings.paths.bm25_dir
    manifest_path = settings.paths.index_dir / "index_manifest.json"
    stats_output = (
        args.stats_output.expanduser().resolve()
        if args.stats_output is not None
        else settings.paths.index_dir / "build_stats.json"
    )

    logger.info(
        "开始构建索引：input=%s, chroma_dir=%s, bm25_dir=%s, collection=%s, rebuild=%s",
        input_path,
        chroma_dir,
        bm25_dir,
        args.collection_name,
        args.rebuild,
    )

    try:
        ensure_dir(settings.paths.index_dir)
        ensure_dir(chroma_dir)
        ensure_dir(bm25_dir)

        chunks, stats = load_kb_chunks(input_path)
        deduped_chunks = deduplicate_chunks(chunks, stats)

        source_type_counter = Counter(chunk.source_type for chunk in deduped_chunks)
        category_counter = Counter(chunk.category or "未分类" for chunk in deduped_chunks)
        stats.by_source_type = dict(sorted(source_type_counter.items()))
        stats.by_category = dict(sorted(category_counter.items()))

        client = get_chroma_client(chroma_dir)
        collection = get_or_reset_collection(
            client=client,
            collection_name=args.collection_name,
            rebuild=args.rebuild,
        )

        manifest = (
            IndexManifest(collection_name=args.collection_name, updated_at=now_iso())
            if args.rebuild
            else load_manifest(manifest_path=manifest_path, collection_name=args.collection_name)
        )

        incremental_candidates = (
            deduped_chunks if args.rebuild else filter_new_chunks_by_manifest(deduped_chunks, manifest, stats)
        )
        stats.chroma_new_candidates = len(incremental_candidates)

        chroma_query_batch_size = min(
            max(args.batch_size, 1),
            get_chroma_max_batch_size(client, fallback=max(args.batch_size, 1)),
        )

        chroma_new_chunks = filter_existing_doc_ids_in_chroma(
            collection=collection,
            chunks=incremental_candidates,
            query_batch_size=chroma_query_batch_size,
            stats=stats,
        )

        embedder = EmbeddingClient(
            model=settings.models.embed_model,
            batch_size=args.batch_size,
        )

        inserted_chunks = upsert_chunks_to_chroma(
            collection=collection,
            chunks=chroma_new_chunks,
            embedder=embedder,
            batch_size=chroma_query_batch_size,
            stats=stats,
        )

        # 保存/更新 manifest
        if args.rebuild:
            manifest.records = []
        append_manifest_records(manifest, inserted_chunks)
        save_manifest(manifest, manifest_path)
        stats.final_manifest_records = len(manifest.records)

        # 构建 BM25：
        # - rebuild：完全基于当前 deduped_chunks 重建
        # - incremental：加载已有语料，再追加本次真正新增到索引的 chunk
        existing_bm25_entries = {} if args.rebuild else load_existing_bm25_entries(bm25_dir / "bm25_corpus.jsonl")

        merged_bm25_entries: dict[str, BM25Entry] = {}
        for doc_id, entry in existing_bm25_entries.items():
            merged_bm25_entries[doc_id] = entry

        bm25_new_docs = 0
        bm25_source_chunks = deduped_chunks if args.rebuild else inserted_chunks
        for chunk in bm25_source_chunks:
            entry = build_bm25_entry(chunk)
            if chunk.doc_id not in merged_bm25_entries:
                bm25_new_docs += 1
            merged_bm25_entries[chunk.doc_id] = entry

        final_bm25_entries = sorted(
            merged_bm25_entries.values(),
            key=lambda item: (item.source_type, item.source_file, item.title, item.doc_id),
        )
        save_bm25_artifacts(final_bm25_entries, bm25_dir=bm25_dir)

        stats.bm25_total_docs = len(final_bm25_entries)
        stats.bm25_new_docs = bm25_new_docs

        write_stats(stats, stats_output)

        logger.info(
            "索引构建完成：input_valid=%s, deduped=%s, chroma_upserted=%s, embedding_success=%s, "
            "embedding_failed=%s, bm25_total=%s, stats=%s",
            stats.input_valid,
            len(deduped_chunks),
            stats.chroma_upserted,
            stats.embedding_success,
            stats.embedding_failed,
            stats.bm25_total_docs,
            stats_output,
        )
        return 0

    except Exception as exc:
        logger.exception("构建索引失败：%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
