from __future__ import annotations

import traceback
from pprint import pprint

from app.logging_utils import setup_logging
from app.clients.llm_client import LLMClient


def main() -> None:
    log_path = setup_logging("INFO", module_name=__name__)
    print(f"日志文件: {log_path}")

    print("=" * 100)
    print("最小 LLM 调用测试开始")
    print("=" * 100)

    # 这里构造最小 messages，不依赖检索、不依赖 PromptBuilder
    messages = [
        {
            "role": "system",
            "content": "你是企业网银 FAQ 助手，请使用简洁中文回答。",
        },
        {
            "role": "user",
            "content": "请用一句话回答：代发工资失败了在哪里看？",
        },
    ]

    print("\n[1] 即将发送给 LLMClient.ask() 的 messages：")
    pprint(messages, sort_dicts=False)

    try:
        # 默认假设你的 LLMClient 支持无参初始化
        client = LLMClient()

        print("\n[2] 开始调用 LLMClient.ask() ...")
        answer = client.ask(messages=messages, temperature=0.2)

        print("\n[3] 模型返回结果：")
        print("-" * 100)
        print(answer)
        print("-" * 100)

        if not answer or not str(answer).strip():
            print("\n测试执行成功，但返回内容为空。")
        else:
            print("\n最小 LLM 调用测试通过。")

    except Exception as exc:
        print("\n最小 LLM 调用测试失败")
        print(f"异常类型: {type(exc).__name__}")
        print(f"异常信息: {exc}")
        print("\n详细堆栈：")
        traceback.print_exc()


if __name__ == "__main__":
    main()