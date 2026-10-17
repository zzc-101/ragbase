"""命令行多轮对话：Agent 按需检索知识库并带 [n] 引用回答（与 POST /chat 同一套 Agent，进程内运行）。

运行：uv run chat.py [--source 书名.pdf ...]
输入问题即可；/new 开始新会话，exit 退出。会话历史只在本进程内存里。
"""
import argparse
import json
import uuid

from dotenv import load_dotenv

from agent_tools import ToolRetrievalError
from chat_agent import ChatService, build_chat_service
from config import get_settings
from hybrid_retriever import UnknownSourceError
from query import source_label
from rag_common import collection_count
from rag_pipeline import GenerationError, build_pipeline

load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(description="RAG 命令行多轮对话（Agent）")
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="NAME",
        help="默认只在这本书里检索（文件名，可重复指定）；模型调用工具时没有另外指定书名就用它；不指定则检索全部",
    )
    return parser.parse_args()


def _args_text(args: dict) -> str:
    return ", ".join(f"{key}={json.dumps(value, ensure_ascii=False)}" for key, value in args.items())


def run_turn(service: ChatService, session_id: str, message: str, sources: list[str] | None) -> None:
    """发送一条消息：实时打印工具调用和回答文本，结束后打印编号来源。出错只打印提示，会话保持可用（失败的那一轮已回滚）。"""
    started, citations = False, []
    try:
        for event, data in service.stream(session_id, message, sources):
            if event == "tool_call":
                print(f"  [工具] {data['name']}({_args_text(data['args'])})")
            elif event == "token":
                if not started:
                    print("\n答：", end="")
                    started = True
                print(data["text"], end="", flush=True)
            elif event == "citations":
                citations = data["citations"]
        print()
    except GenerationError as exc:
        print(f"\n错误：{'大模型响应超时' if exc.timeout else '大模型调用失败'}，请稍后重试（{exc}）")
        return
    except ToolRetrievalError as exc:
        print(f"\n错误：{exc}")
        return
    if citations:
        print("\n来源：")
        for number, doc in citations:
            print(f"[{number}] {source_label(doc.metadata)}")


def main():
    args = parse_args()
    settings = get_settings()
    if not settings.llm_configured:
        print("错误：请先在 .env 里配置 RAG_LLM_API_KEY、RAG_LLM_MODEL 和 RAG_LLM_BASE_URL（旧的 ARK_* 变量同样有效）")
        return
    if not collection_count(settings.db_dir):
        print("错误：索引不存在或为空，请先运行：uv run ingest.py")
        return

    print("加载本地 embedding 与 reranker 模型...")
    pipeline, _ = build_pipeline(settings=settings, build_bm25=True)
    sources = list(dict.fromkeys(args.source)) or None
    if sources:
        try:
            pipeline.retriever.resolve_sources(sources)
        except UnknownSourceError as exc:
            print(f"错误：未知的书名 {exc.unknown}")
            print("可选：" + "、".join(exc.valid))
            return
    service = build_chat_service(pipeline, settings)

    scope = f"，默认限定：{'、'.join(sources)}" if sources else ""
    print(f"对话已就绪{scope}。输入问题开始；/new 新会话，exit 退出")
    session_id = uuid.uuid4().hex
    while True:
        try:
            message = input("\n问：").strip()
        except EOFError:
            break
        if message.lower() in ("exit", "quit", "q"):
            print("再见！")
            break
        if message == "/new":
            service.delete(session_id)
            session_id = uuid.uuid4().hex
            print("已开始新会话")
            continue
        if message:
            run_turn(service, session_id, message[:2000], sources)  # 与 POST /chat 的 message 上限一致


if __name__ == "__main__":
    main()
