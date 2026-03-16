from __future__ import annotations

from pathlib import Path
import traceback

from app.retrieval.bm25_index import BM25Index

BM25_DIR = Path("../../data/index/bm25")

TEST_QUERIES = [
    "代发工资失败了在哪里看",
    "UKey 插上没反应怎么办",
    "用户暂无权限怎么办",
    "回单红叉怎么办",
    "undefined message 报错怎么办",
]


def print_result_item(item, idx: int) -> None:
    print(f"\n--- result #{idx} ---")
    if isinstance(item, dict):
        print("doc_id:", item.get("doc_id"))
        print("title:", item.get("title"))
        print("score:", item.get("score"))
        print("source_type:", item.get("source_type"))
        print("category:", item.get("category"))
    else:
        # 如果你的 search 返回的是自定义对象，这里直接打印
        print(item)


def build_index() -> BM25Index:
    """
    尝试兼容几种常见初始化方式：
    1. BM25Index()
    2. BM25Index(index_dir=...)
    3. BM25Index.load(...)
    """
    # 方式 1：无参初始化
    try:
        return BM25Index()
    except TypeError:
        pass

    # 方式 2：传 index_dir
    try:
        return BM25Index(index_dir=BM25_DIR)
    except TypeError:
        pass

    # 方式 3：类方法 load
    if hasattr(BM25Index, "load"):
        return BM25Index.load(BM25_DIR)

    raise RuntimeError("无法确定 BM25Index 的初始化方式，请检查 app/retrieval/bm25_index.py。")


def main() -> None:
    print("=" * 80)
    print("Step 1 | BM25 索引目录检查")
    print("=" * 80)
    print(f"BM25 目录: {BM25_DIR.resolve()}")
    print(f"目录是否存在: {BM25_DIR.exists()}")

    if not BM25_DIR.exists():
        raise FileNotFoundError(f"BM25 目录不存在: {BM25_DIR}")

    print("\n" + "=" * 80)
    print("Step 2 | 初始化 BM25Index")
    print("=" * 80)
    index = build_index()
    print(f"BM25Index 初始化成功: {type(index).__name__}")

    print("\n" + "=" * 80)
    print("Step 3 | 执行若干典型搜索")
    print("=" * 80)

    for query in TEST_QUERIES:
        print(f"\n>>> query = {query}")
        results = index.search(query, top_k=3)
        print(f"返回结果数: {len(results)}")

        if not results:
            print("⚠未返回结果")
            continue

        for i, item in enumerate(results[:3], start=1):
            print_result_item(item, i)

    print("\nBM25 可加载性测试通过。")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\nBM25 可加载性测试失败")
        print(f"异常类型: {type(exc).__name__}")
        print(f"异常信息: {exc}")
        print("\n详细堆栈:")
        traceback.print_exc()
