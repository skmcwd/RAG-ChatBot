from __future__ import annotations

import traceback

from app.clients.embedding_client import EmbeddingClient
from app.retrieval.vector_store import VectorStore


def main() -> None:
    query = "代发工资失败了在哪里看"
    print(f"测试查询: {query}")

    embedder = EmbeddingClient(batch_size=1)
    store = VectorStore()

    query_embedding = embedder.embed_text(query)
    print(f"query embedding 维度: {len(query_embedding)}")

    results = store.query_by_embedding(query_embedding, top_k=5)
    print(f"返回结果数: {len(results)}")

    for i, item in enumerate(results, start=1):
        print(f"\n--- result #{i} ---")
        if isinstance(item, dict):
            print("doc_id:", item.get("doc_id"))
            print("title:", item.get("metadata", {}).get("title"))
            print("source_type:", item.get("metadata", {}).get("source_type"))
            print("distance/similarity:", item.get("distance") or item.get("score"))
        else:
            print(item)

    print("\nChroma query_by_embedding 测试通过。")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\nChroma query_by_embedding 测试失败")
        print(type(exc).__name__, exc)
        traceback.print_exc()
