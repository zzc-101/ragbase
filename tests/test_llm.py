"""大模型工厂、超时 / 重试和 /ask 端到端测试：全部走本机 127.0.0.1 的 OpenAI 兼容桩服务，不访问外网、不用真实密钥。

覆盖：create_chat_model 的参数透传与未配置 / 协议错误；回答链解析【依据】/【回答】；
5xx 自动重试（数请求次数）；超时 -> GenerationError(timeout=True)；/ask 的 200 / 504 / 502。

运行：uv run python tests/test_llm.py
"""
import json
import logging
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from langchain_core.documents import Document

import api
from config import Settings
from llm_factory import create_chat_model
from rag_pipeline import GenerationError, RAGPipeline, ask, build_answer_chain

REPLY = "【依据】\n- [片段1] “挣值分析用于衡量项目绩效”\n【回答】\n挣值分析可以衡量项目的进度与成本绩效 [片段1]"
SECRET = "sk-stub-SECRET-123"


class StubLLM:
    """本机 OpenAI 兼容桩：POST /v1/chat/completions，非流式。

    fail_first：前 N 次请求返回 HTTP 500；delay：每次响应前先睡这么久（秒，用于模拟超时）。
    requests 记录每次到达的 (请求头, JSON 体)。
    """

    def __init__(self, reply=REPLY, fail_first=0, delay=0.0):
        self.reply, self.fail_first, self.delay = reply, fail_first, delay
        self.requests = []
        self._lock = threading.Lock()
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # 不往 stderr 打访问日志
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                with stub._lock:
                    stub.requests.append((dict(self.headers), body, self.path))
                    number = len(stub.requests)
                try:
                    if stub.delay:
                        time.sleep(stub.delay)
                    if number <= stub.fail_first:
                        self._send(500, {"error": {"message": "stub boom", "type": "server_error"}})
                        return
                    self._send(200, {
                        "id": "chatcmpl-stub", "object": "chat.completion", "created": 0, "model": body.get("model"),
                        "choices": [{"index": 0, "finish_reason": "stop",
                                     "message": {"role": "assistant", "content": stub.reply}}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                    })
                except (BrokenPipeError, ConnectionResetError):  # 客户端已超时断开
                    pass

            def _send(self, status, payload):
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def __enter__(self):
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
        return self

    def __exit__(self, *exc_info):
        self.server.shutdown()
        self.server.server_close()

    @property
    def count(self) -> int:
        return len(self.requests)


def _settings(stub: StubLLM, **overrides) -> Settings:
    values = dict(
        llm_model="stub-model", llm_api_key=SECRET, llm_base_url=stub.base_url,
        llm_timeout=5, llm_max_retries=0, device="cpu", log_level="WARNING",
    )
    return Settings(_env_file=None, **{**values, **overrides})


class _Retriever:
    bm25_ready = True

    def catalog(self):
        return [{"name": "book.pdf", "chunk_count": 2, "heading_chunks": 0, "has_headings": False}]

    def resolve_sources(self, names):
        return [f"/data/{name}" for name in names]

    def invoke(self, question, mode=None, sources=None):
        return [
            Document(page_content=f"挣值分析用于衡量项目绩效{i}", metadata={
                "chunk_id": f"c{i}", "source": "/data/book.pdf", "page": i, "vec_rank": i + 1, "bm25_rank": None,
            })
            for i in range(2)
        ]


class _Reranker:
    def rerank_with_scores(self, query, documents, top_k=5):
        return [(0, 0.99), (1, 0.98)][:top_k]


def _pipeline(settings: Settings) -> RAGPipeline:
    """假检索 + 假精排 + 真实的回答链（经 llm_factory 指向桩服务）。"""
    return RAGPipeline(
        retriever=_Retriever(), reranker=_Reranker(), embeddings=None, device="cpu",
        answer_chain=build_answer_chain(create_chat_model(settings)),
    )


# ---------------------------------------------------------------------------
# 工厂
# ---------------------------------------------------------------------------

def test_factory_returns_none_when_not_configured_and_rejects_unknown_protocol():
    unset = dict(llm_model=None, llm_api_key=None, llm_base_url=None)
    assert create_chat_model(Settings(_env_file=None, **unset)) is None
    assert create_chat_model(Settings(_env_file=None, **unset), "eval") is None
    try:
        Settings(_env_file=None, llm_protocol="anthropic")
    except ValueError:
        pass
    else:
        raise AssertionError("配置层应拒绝未知协议")
    forced = Settings.model_construct(llm_protocol="anthropic", llm_model="m", llm_api_key=None, llm_base_url=None,
                                      llm_temperature=0, llm_max_tokens=None, llm_timeout=1, llm_max_retries=0,
                                      eval_llm_model=None, eval_llm_api_key=None, eval_llm_base_url=None)
    for purpose in ("answer", "eval"):
        try:
            create_chat_model(forced, purpose)  # 没配齐也要先报协议错误：启动阶段就失败
        except ValueError as exc:
            assert "anthropic" in str(exc) and "openai" in str(exc), exc
        else:
            raise AssertionError("工厂应拒绝未知协议")


def test_factory_passes_settings_to_client_and_request():
    with StubLLM() as stub:
        settings = _settings(stub, llm_max_tokens=77, llm_temperature=0.25, llm_timeout=12.5, llm_max_retries=4)
        llm = create_chat_model(settings)
        assert (llm.request_timeout, llm.max_retries, llm.temperature, llm.max_tokens) == (12.5, 4, 0.25, 77)
        assert SECRET not in repr(llm) and SECRET not in str(llm.openai_api_key)  # 密钥不进 repr
        assert llm.invoke("你好").content == REPLY
        headers, body, path = stub.requests[0]
        assert path == "/v1/chat/completions" and headers["Authorization"] == f"Bearer {SECRET}"
        assert body["model"] == "stub-model" and body["temperature"] == 0.25
        assert 77 in (body.get("max_tokens"), body.get("max_completion_tokens")), body


def test_factory_eval_purpose_uses_override_model():
    with StubLLM() as stub:
        settings = _settings(stub, eval_llm_model="judge-model")
        create_chat_model(settings, "eval").invoke("评分")
        create_chat_model(settings, "answer").invoke("回答")
        assert [body["model"] for _, body, _ in stub.requests] == ["judge-model", "stub-model"]


def test_ragas_llm_uses_eval_model_and_client_limits():
    from llm_factory import create_ragas_llm

    settings = Settings(_env_file=None, llm_model="main", llm_api_key=SECRET, llm_base_url="http://127.0.0.1:9/v1",
                        eval_llm_model="judge", llm_timeout=7, llm_max_retries=1)
    llm = create_ragas_llm(settings)  # 只构造客户端，不发请求
    assert llm.model == "judge" and llm.client.timeout == 7 and llm.client.max_retries == 1
    unset = Settings(_env_file=None, llm_model=None, llm_api_key=None, llm_base_url=None)
    assert create_ragas_llm(unset) is None


# ---------------------------------------------------------------------------
# 回答链解析、重试、超时
# ---------------------------------------------------------------------------

def test_answer_chain_parses_evidence_and_answer():
    with StubLLM() as stub:
        settings = _settings(stub)
        result = ask(_pipeline(settings), "什么是挣值分析？")
        assert not result.refused and stub.count == 1
        assert result.evidence == "- [片段1] “挣值分析用于衡量项目绩效”"
        assert result.answer == "挣值分析可以衡量项目的进度与成本绩效 [片段1]"
        assert len(result.docs) == 2
        messages = stub.requests[0][1]["messages"]
        assert "[片段1]" in messages[0]["content"] and "什么是挣值分析" in messages[1]["content"]


def test_retries_on_server_errors_then_succeeds():
    with StubLLM(fail_first=2) as stub:
        started = time.perf_counter()
        result = ask(_pipeline(_settings(stub, llm_max_retries=2)), "q")
        elapsed = time.perf_counter() - started
        assert stub.count == 3, stub.count  # 失败 2 次 + 成功 1 次
        print(f"  重试：共 {stub.count} 次请求，用时 {elapsed:.2f}s")
        assert result.answer.startswith("挣值分析")
        assert elapsed < 8, elapsed  # 指数退避的上限远小于此


def test_gives_up_after_max_retries_with_generation_error():
    with StubLLM(fail_first=99) as stub:
        try:
            ask(_pipeline(_settings(stub, llm_max_retries=1)), "q")
        except GenerationError as exc:
            assert exc.timeout is False and "超时" not in str(exc)
        else:
            raise AssertionError("重试用完仍失败应抛 GenerationError")
        assert stub.count == 2, stub.count  # 1 次 + 1 次重试


def test_timeout_raises_generation_error_within_bound():
    timeout = 0.5
    with StubLLM(delay=3) as stub:
        started = time.perf_counter()
        try:
            ask(_pipeline(_settings(stub, llm_timeout=timeout, llm_max_retries=0)), "q")
        except GenerationError as exc:
            assert exc.timeout is True and "超时" in str(exc), exc
        else:
            raise AssertionError("应抛 GenerationError")
        elapsed = time.perf_counter() - started
        print(f"  超时 {timeout}s / 不重试：{stub.count} 次请求，{elapsed:.2f}s 后抛出")
        assert stub.count == 1 and timeout <= elapsed < timeout * 1 + 1.5, (stub.count, elapsed)


def test_timeout_is_retried_before_giving_up():
    timeout = 0.5
    with StubLLM(delay=3) as stub:
        started = time.perf_counter()
        try:
            ask(_pipeline(_settings(stub, llm_timeout=timeout, llm_max_retries=1)), "q")
        except GenerationError as exc:
            assert exc.timeout is True
        else:
            raise AssertionError("应抛 GenerationError")
        elapsed = time.perf_counter() - started
        print(f"  超时 {timeout}s / 重试 1 次：{stub.count} 次请求，{elapsed:.2f}s 后抛出")
        assert stub.count == 2, stub.count
        assert elapsed < timeout * 2 + 0.5 + 1.5, elapsed  # 2 次超时 + 至多约 0.5s 退避 + 余量


# ---------------------------------------------------------------------------
# API：/ask 端到端（真实回答链 -> 桩服务）
# ---------------------------------------------------------------------------

def _ask_via_api(settings: Settings, body=None):
    original = api.collection_count
    api.collection_count = lambda *args, **kwargs: 2
    logging.disable(logging.CRITICAL)  # API 会 logger.exception 整个调用栈，测试输出里不需要
    try:
        app = api.create_app(pipeline_factory=lambda _s: _pipeline(settings), settings=settings)
        with TestClient(app) as client:
            health = client.get("/health").json()
            return client.post("/ask", json=body or {"query": "什么是挣值分析？"}), health
    finally:
        logging.disable(logging.NOTSET)
        api.collection_count = original


def test_api_ask_end_to_end_success():
    with StubLLM() as stub:
        response, health = _ask_via_api(_settings(stub))
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["refused"] is False
        assert body["evidence"] == "- [片段1] “挣值分析用于衡量项目绩效”"
        assert body["answer"] == "挣值分析可以衡量项目的进度与成本绩效 [片段1]"
        assert [c["n"] for c in body["citations"]] == [1, 2] and body["citations"][0]["source"] == "book.pdf"
        assert health["llm_configured"] is True and health["llm_protocol"] == "openai"
        assert health["llm_model"] == "stub-model" and SECRET not in json.dumps(health)
        assert stub.count == 1


def test_api_ask_timeout_returns_504_unified_error():
    with StubLLM(delay=3) as stub:
        started = time.perf_counter()
        response, _ = _ask_via_api(_settings(stub, llm_timeout=0.5, llm_max_retries=0))
        elapsed = time.perf_counter() - started
    assert response.status_code == 504, (response.status_code, response.text)
    body = response.json()
    assert body == {"error": "llm_timeout", "detail": "大模型响应超时，请稍后重试"}, body
    assert SECRET not in response.text and "127.0.0.1" not in response.text
    assert elapsed < 4, elapsed


def test_api_ask_server_error_returns_502_unified_error():
    with StubLLM(fail_first=99) as stub:
        response, _ = _ask_via_api(_settings(stub, llm_max_retries=0))
        assert stub.count == 1
    assert response.status_code == 502, (response.status_code, response.text)
    assert response.json() == {"error": "llm_error", "detail": "大模型调用失败，请稍后重试"}
    assert SECRET not in response.text and not re.search(r"500|stub boom", response.text)


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    started = time.perf_counter()
    for test in tests:
        test()
        print(f"通过：{test.__name__}")
    print(f"共 {len(tests)} 项测试通过（{time.perf_counter() - started:.1f}s）")
