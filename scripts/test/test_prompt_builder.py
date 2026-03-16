from __future__ import annotations

import traceback
from pprint import pprint

from app.logging_utils import setup_logging
from app.models import RetrievedChunk
from app.services.prompt_builder import build_messages


def main() -> None:
    log_path = setup_logging("INFO", module_name=__name__)
    print(f"日志文件: {log_path}")

    print("=" * 100)
    print("PromptBuilder 构造测试开始")
    print("=" * 100)

    user_query = "代发工资失败了在哪里看？"

    # 注意：这里必须构造 RetrievedChunk 对象，而不是 dict
    retrieved_chunks = [
        RetrievedChunk(
            doc_id="docx_chunk_001",
            source_file="操作手册-如何网银代发工资.docx",
            source_type="docx",
            title="代发工资结果查询路径",
            category="代发",
            question=None,
            answer=None,
            full_text="代发工资交易结果可通过企业网银相关代发工资查询功能进行查看，也可结合柜面交易结果进行核对。",
            keywords=["代发工资", "结果查询", "企业网银"],
            image_paths=[],
            page_no=1,
            slide_no=None,
            priority=1.1,
            chunk_hash="dummy_hash_001",
            retrieval_score=1.25,
            vector_score=0.82,
            bm25_score=8.13,
            rerank_reason="命中代发+查询语义，且为手册型高优先级来源",
        ),
        RetrievedChunk(
            doc_id="ppt_slide_012",
            source_file="企业网银问题带图.pptx",
            source_type="ppt",
            title="代发失败查询",
            category="代发",
            question=None,
            answer=None,
            full_text="若代发业务提交后显示失败，可进入代发结果查询页面查看失败明细，并按页面提示核对模板、字段与账户信息。",
            keywords=["代发失败", "结果查询", "模板"],
            image_paths=["data/parsed/images/ppt_slide_012_img_1.png"],
            page_no=None,
            slide_no=12,
            priority=0.95,
            chunk_hash="dummy_hash_002",
            retrieval_score=1.10,
            vector_score=0.79,
            bm25_score=7.55,
            rerank_reason="命中代发失败查询问题，且含图文说明",
        ),
    ]

    print("\n[1] 用户问题：")
    print(user_query)

    print("\n[2] 输入给 build_messages() 的 retrieved_chunks：")
    pprint([item.model_dump() for item in retrieved_chunks], sort_dicts=False)

    try:
        print("\n[3] 开始调用 build_messages() ...")
        messages = build_messages(
            user_query=user_query,
            retrieved_chunks=retrieved_chunks,
        )

        print("\n[4] build_messages() 返回结果：")
        print("-" * 100)
        pprint(messages, sort_dicts=False)
        print("-" * 100)

        print(f"\n消息条数: {len(messages)}")
        for i, msg in enumerate(messages, start=1):
            role = msg.get("role", "-")
            content = msg.get("content", "")
            content_preview = str(content).replace("\n", " ").strip()
            if len(content_preview) > 250:
                content_preview = content_preview[:250] + "..."
            print(f"\n消息 #{i}")
            print(f"role   : {role}")
            print(f"preview: {content_preview}")

        print("\nPromptBuilder 构造测试通过。")

    except Exception as exc:
        print("\nPromptBuilder 构造测试失败")
        print(f"异常类型: {type(exc).__name__}")
        print(f"异常信息: {exc}")
        print("\n详细堆栈：")
        traceback.print_exc()


if __name__ == "__main__":
    main()
