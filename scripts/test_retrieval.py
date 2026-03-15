from __future__ import annotations

from pathlib import Path
import traceback
from typing import Any

from app.logging_utils import setup_logging
from app.retrieval.hybrid_retriever import HybridRetriever


TEST_QUERIES = [
    "代发工资失败了在哪里看？",
    "UKey 插上没反应怎么办？",
    "用户暂无权限怎么办？",
    "电子回单印章显示红叉怎么办？",
    "为什么每次登录都要重新下载控件？",
    "代发工资报 undefined message 怎么办？",
    "代发工资的交易入口在哪里？",
    "证书初始化或证书下载相关问题怎么处理？",
]


def safe_get(obj: Any, key: str, default: Any = None) -> Any:
    """
    兼容 dict / pydantic model / 普通对象 三种取值方式。
    """
    if obj is None:
        return default

    if isinstance(obj, dict):
        return obj.get(key, default)

    if hasattr(obj, key):
        return getattr(obj, key, default)

    # 某些 pydantic/model_dump 结构可选
    if hasattr(obj, "model_dump"):
        try:
            data = obj.model_dump()
            if isinstance(data, dict):
                return data.get(key, default)
        except Exception:
            pass

    return default


def to_dict(obj: Any) -> dict[str, Any]:
    """
    尽量把返回对象转成 dict，便于统一打印。
    """
    if obj is None:
        return {}

    if isinstance(obj, dict):
        return obj

    if hasattr(obj, "model_dump"):
        try:
            data = obj.model_dump()
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    if hasattr(obj, "__dict__"):
        try:
            return dict(obj.__dict__)
        except Exception:
            pass

    return {"raw_object": repr(obj)}


def extract_retrieval_payload(result: Any) -> tuple[list[Any], dict[str, Any]]:
    """
    兼容不同 retrieve() 返回形态。

    约定：
    1. 若 retrieve() 直接返回 list，则认为它就是结果列表，debug_info 为空。
    2. 若 retrieve() 返回 dict，则优先尝试以下 key：
       - results
       - retrieved_chunks
       - items
       同时提取 debug_info。
    3. 若 retrieve() 返回对象，则尝试属性：
       - results
       - retrieved_chunks
       - items
       同时提取 debug_info。
    """
    # 直接返回 list
    if isinstance(result, list):
        return result, {}

    # dict 形式
    if isinstance(result, dict):
        results = (
                result.get("results")
                or result.get("retrieved_chunks")
                or result.get("items")
                or []
        )
        debug_info = result.get("debug_info", {}) or {}
        return results, debug_info

    # 对象形式
    results = (
            safe_get(result, "results")
            or safe_get(result, "retrieved_chunks")
            or safe_get(result, "items")
            or []
    )
    debug_info = safe_get(result, "debug_info", {}) or {}
    return results, debug_info


def format_score(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def format_bool(value: Any) -> str:
    return "是" if bool(value) else "否"


def print_result_item(item: Any, rank: int) -> None:
    """
    打印单条检索结果。
    """
    data = to_dict(item)

    metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    source_file = (
            data.get("source_file")
            or metadata.get("source_file")
            or "-"
    )
    source_type = (
            data.get("source_type")
            or metadata.get("source_type")
            or "-"
    )
    category = (
            data.get("category")
            or metadata.get("category")
            or "-"
    )
    title = (
            data.get("title")
            or metadata.get("title")
            or data.get("doc_id")
            or "-"
    )
    doc_id = data.get("doc_id") or metadata.get("doc_id") or "-"
    retrieval_score = data.get("retrieval_score") or data.get("final_score") or data.get("score")
    vector_score = data.get("vector_score")
    bm25_score = data.get("bm25_score")
    rerank_reason = data.get("rerank_reason") or "-"
    image_paths = (
            data.get("image_paths")
            or metadata.get("image_paths")
            or []
    )
    has_image = bool(image_paths) or bool(data.get("has_image")) or bool(metadata.get("has_image"))

    slide_no = data.get("slide_no") or metadata.get("slide_no")
    page_no = data.get("page_no") or metadata.get("page_no")

    print(f"\n  [{rank}] {title}")
    print(f"      doc_id          : {doc_id}")
    print(f"      source_file     : {source_file}")
    print(f"      source_type     : {source_type}")
    print(f"      category        : {category}")
    print(f"      retrieval_score : {format_score(retrieval_score)}")
    print(f"      vector_score    : {format_score(vector_score)}")
    print(f"      bm25_score      : {format_score(bm25_score)}")
    print(f"      has_image       : {format_bool(has_image)}")
    print(f"      slide_no/page_no: {slide_no if slide_no is not None else '-'} / {page_no if page_no is not None else '-'}")
    print(f"      rerank_reason   : {rerank_reason}")

    # 简短预览正文
    preview = data.get("full_text") or data.get("document") or data.get("text") or ""
    if preview:
        preview = str(preview).replace("\n", " ").strip()
        if len(preview) > 120:
            preview = preview[:120] + "..."
        print(f"      preview         : {preview}")


def print_debug_info(debug_info: dict[str, Any]) -> None:
    """
    打印关键 debug 信息。
    """
    if not debug_info:
        print("  debug_info: <空>")
        return

    normalized_query = debug_info.get("normalized_query")
    expanded_terms = debug_info.get("expanded_terms")
    guessed_categories = debug_info.get("guessed_categories")
    exact_terms = debug_info.get("exact_terms")

    print("  debug_info:")
    print(f"    normalized_query : {normalized_query or '-'}")
    print(f"    expanded_terms   : {expanded_terms or []}")
    print(f"    guessed_categories: {guessed_categories or []}")
    print(f"    exact_terms      : {exact_terms or []}")


def main() -> None:
    log_path = setup_logging("INFO")
    print(f"日志文件: {log_path}")

    print("=" * 100)
    print("开始执行检索层单独验收：HybridRetriever.retrieve()")
    print("=" * 100)

    retriever = HybridRetriever()

    total_queries = len(TEST_QUERIES)
    passed_queries = 0

    for idx, query in enumerate(TEST_QUERIES, start=1):
        print("\n" + "=" * 100)
        print(f"测试问题 {idx}/{total_queries}")
        print(f"原始问题: {query}")
        print("=" * 100)

        try:
            result = retriever.retrieve(query)
            results, debug_info = extract_retrieval_payload(result)

            print_debug_info(debug_info)

            if not results:
                print("\n 未返回任何检索结果")
                continue

            print(f"\n  返回结果数: {len(results)}")
            print("  Top 3 结果概览：")

            for rank, item in enumerate(results[:3], start=1):
                print_result_item(item, rank)

            # 统计 top3 中是否命中图片
            top3 = results[:3]
            top3_has_image = any(
                bool(
                    safe_get(item, "image_paths")
                    or safe_get(safe_get(item, "metadata", {}), "image_paths")
                    or safe_get(item, "has_image")
                    or safe_get(safe_get(item, "metadata", {}), "has_image")
                )
                for item in top3
            )
            print(f"\n  Top3 是否命中图片证据: {format_bool(top3_has_image)}")
            print(" 本问题检索执行完成")
            passed_queries += 1

        except Exception as exc:
            print(f"\n 检索执行失败: {type(exc).__name__}: {exc}")
            traceback.print_exc()

    print("\n" + "=" * 100)
    print("检索层验收结束")
    print(f"成功执行问题数: {passed_queries}/{total_queries}")
    print("=" * 100)


if __name__ == "__main__":
    main()