"""模型提供方（本地 / 云端 embedding 与 reranker）的测试：全部走本机 127.0.0.1 的桩服务，不访问外网、不用真实密钥、不加载本地模型。

覆盖：
- OpenAI 兼容 embedding：分批、dimensions 参数、L2 归一化、Authorization 头、乱序 index、查询指令前缀（只加查询）、查询缓存、
  重试 / 错误信息（不含密钥）/ 向量数量或维度不符；
- rerank API：请求形状、top_n、按 index 映射分数、忽略 top_n 的服务、错误；
- provider=none：不精排、分数为 None、不按得分拒答（pipeline / ask / HTTP 接口）；
- 配置与工厂：未知提供方、密钥回退到大模型配置（及“不同地址不沿用密钥”）、密钥不进 repr、地址缺失报错、
  models download / doctor / health 的行为；
- 全云端配置不依赖 torch（子进程里把 torch 等标记为不可导入，仍能入库 + 检索）；
- 索引签名守卫：匹配 / 不一致拒绝 / 旧索引补记 / --full 改写 / 向量长度校验 / 命令行端到端。

运行：uv run python tests/test_providers.py
"""
import contextlib
import io
import json
import logging
import os
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from fastapi.testclient import TestClient
from langchain_core.documents import Document

from ragbase import cli, config, doctor, index_info
from ragbase.api import app as api
from ragbase.config import Settings, get_settings
from ragbase.index_info import EmbeddingSignature, IndexMismatchError, IndexRecorder
from ragbase.models import api_client, download
from ragbase.models.api_client import ApiClient, ProviderError
from ragbase.models.api_embeddings import OpenAIEmbeddings
from ragbase.models.api_reranker import ApiReranker, NoReranker
from ragbase.models.factory import ProviderConfigError, create_embeddings, create_reranker, provider_problems
from ragbase.retrieval.pipeline import NO_ANSWER, RAGPipeline, ask, retrieve_and_rerank, select_context_docs, should_refuse

ROOT = Path(__file__).resolve().parent.parent
KEY = "sk-stub-EMBED-KEY-111"
LLM_KEY = "sk-stub-LLM-KEY-222"
_SCRATCH = tempfile.TemporaryDirectory(prefix="ragdemo-providers-test-")
SCRATCH = Path(_SCRATCH.name).resolve()
_counter = iter(range(10_000))
api_client.BACKOFF_BASE = 0.0  # 重试不真的等待


def _fresh_dir(name: str) -> Path:
    path = SCRATCH / f"{name}-{next(_counter)}"
    path.mkdir()
    return path


# ---------------------------------------------------------------------------
# 桩服务：OpenAI 兼容 /v1/embeddings + 通用 /v1/rerank
# ---------------------------------------------------------------------------

def fake_vector(text: str, size: int = 16) -> list[float]:
    """确定性词袋向量（按字符哈希进桶，不归一化）：字符重叠越多越相似，让入库 + 检索的端到端测试有意义。"""
    vector = [0.0] * size
    for char in text:
        vector[ord(char) % size] += 1.0
    return vector if any(vector) else [1.0] + [0.0] * (size - 1)


class StubProviders:
    """embedding 与 rerank 桩。requests 记录 (请求头, JSON 体, 路径)。

    fail_first：前 N 次请求返回 HTTP 500；status：所有请求都返回该状态码（错误体里会回显请求头里的密钥，用来验证抹除）；
    shuffle：embedding 响应里的 data 倒序返回（靠 index 还原）；honor_dimensions：是否按 dimensions 截断；
    drop_one：少返回一个向量；plain_text：返回非 JSON；delay：响应前睡眠；rerank_all：忽略 top_n 返回全部（升序，测试客户端排序）。
    """

    def __init__(self, fail_first=0, status=None, shuffle=False, honor_dimensions=True, drop_one=False,
                 plain_text=False, delay=0.0, rerank_all=False, vector_size=16):
        self.fail_first, self.status, self.shuffle = fail_first, status, shuffle
        self.honor_dimensions, self.drop_one, self.plain_text = honor_dimensions, drop_one, plain_text
        self.delay, self.rerank_all, self.vector_size = delay, rerank_all, vector_size
        self.requests = []
        self._lock = threading.Lock()
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                with stub._lock:
                    stub.requests.append((dict(self.headers), body, self.path))
                    number = len(stub.requests)
                try:
                    if stub.delay:
                        time.sleep(stub.delay)
                    if stub.status:
                        auth = self.headers.get("Authorization", "")
                        self._send(stub.status, {"error": {"message": f"denied for {auth}"}})
                    elif number <= stub.fail_first:
                        self._send(500, {"error": {"message": "stub boom"}})
                    elif stub.plain_text:
                        self._send_raw(200, b"<html>not json</html>")
                    elif self.path.endswith("/embeddings"):
                        self._send(200, stub._embeddings(body))
                    elif self.path.endswith("/rerank"):
                        self._send(200, stub._rerank(body))
                    else:
                        self._send(404, {"error": "unknown path"})
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def _send(self, status, payload):
                self._send_raw(status, json.dumps(payload).encode())

            def _send_raw(self, status, data):
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.origin = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.base_url = f"{self.origin}/v1"

    def _embeddings(self, body):
        texts = body["input"]
        size = body["dimensions"] if body.get("dimensions") and self.honor_dimensions else self.vector_size
        data = [{"object": "embedding", "index": i, "embedding": fake_vector(t, size)} for i, t in enumerate(texts)]
        if self.drop_one:
            data = data[:-1]
        if self.shuffle:
            data.reverse()
        return {"object": "list", "model": body["model"], "data": data, "usage": {"prompt_tokens": 1, "total_tokens": 1}}

    def _rerank(self, body):
        query, docs = body["query"], body["documents"]
        scored = [{"index": i, "relevance_score": sum(char in doc for char in query) / max(len(query), 1)}
                  for i, doc in enumerate(docs)]
        scored.sort(key=lambda item: item["relevance_score"], reverse=not self.rerank_all)
        if not self.rerank_all:
            scored = scored[: body.get("top_n", len(docs))]
        return {"id": "stub", "results": scored}

    def __enter__(self):
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
        return self

    def __exit__(self, *exc_info):
        self.server.shutdown()
        self.server.server_close()

    @property
    def count(self) -> int:
        return len(self.requests)

    def paths(self) -> list[str]:
        return [path for _, _, path in self.requests]


def _settings(stub: StubProviders | None = None, **overrides) -> Settings:
    values = dict(device="cpu", log_level="WARNING", llm_model=None, llm_api_key=None, llm_base_url=None)
    if stub is not None:
        values.update(
            embedding_provider="openai", embedding_base_url=stub.base_url, embedding_api_key=KEY, embedding_model="stub-embed",
            reranker_provider="api", reranker_base_url=stub.base_url, reranker_api_key=KEY, reranker_model="stub-rerank",
            embedding_max_retries=0, reranker_max_retries=0, embedding_timeout=5, reranker_timeout=5,
        )
    return Settings(_env_file=None, **{**values, **overrides})


@contextlib.contextmanager
def clean_env(**values):
    """临时清掉所有 RAG_* / ARK_* 环境变量再设置给定的，保证与外部环境无关；退出时还原，并清空 get_settings 缓存。"""
    saved = dict(os.environ)
    for key in list(os.environ):
        if key.startswith(("RAG_", "ARK_")):
            del os.environ[key]
    os.environ.update(values)
    get_settings.cache_clear()
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)
        get_settings.cache_clear()


def _raises(exc_type, func, *needles):
    try:
        func()
    except exc_type as exc:
        for needle in needles:
            assert needle in str(exc), (needle, str(exc))
        return exc
    raise AssertionError(f"应抛出 {exc_type.__name__}")


# ---------------------------------------------------------------------------
# embedding
# ---------------------------------------------------------------------------

def _norm(vector):
    return sum(x * x for x in vector) ** 0.5


def test_embeddings_batching_auth_normalization_and_order():
    with StubProviders(shuffle=True) as stub:
        embeddings = create_embeddings(_settings(stub, embedding_batch_size=3))
        texts = [f"文本{i}" * (i + 1) for i in range(7)]
        vectors = embeddings.embed_documents(texts)
    assert len(vectors) == 7
    assert [len(body["input"]) for _, body, _ in stub.requests] == [3, 3, 1]          # 分批
    assert stub.paths() == ["/v1/embeddings"] * 3
    headers, body, _ = stub.requests[0]
    assert headers["Authorization"] == f"Bearer {KEY}" and body["model"] == "stub-embed"
    assert "dimensions" not in body                                                    # 没设置就不发
    for vector, text in zip(vectors, texts):
        assert abs(_norm(vector) - 1.0) < 1e-9                                          # L2 归一化
        raw = fake_vector(text)
        expected = [x / _norm(raw) for x in raw]
        assert all(abs(a - b) < 1e-9 for a, b in zip(vector, expected)), "乱序返回时必须按 index 还原顺序"


def test_embeddings_dimensions_param_and_mismatch():
    with StubProviders() as stub:
        embeddings = create_embeddings(_settings(stub, embedding_dimensions=8))
        vectors = embeddings.embed_documents(["甲", "乙"])
        assert stub.requests[0][1]["dimensions"] == 8 and {len(v) for v in vectors} == {8}
    with StubProviders(honor_dimensions=False) as stub:  # 服务端不支持 dimensions：返回 16 维
        embeddings = create_embeddings(_settings(stub, embedding_dimensions=8))
        _raises(ProviderError, lambda: embeddings.embed_documents(["甲"]), "RAG_EMBEDDING_DIMENSIONS=8", "16")


def test_embeddings_query_instruction_only_for_queries_and_query_cache():
    with StubProviders() as stub:
        embeddings = create_embeddings(_settings(stub, embedding_query_instruction="Instruct: 检索\\nQuery: "))
        assert embeddings.query_instruction == "Instruct: 检索\nQuery: "              # \n 还原成换行
        first = embeddings.embed_query("什么是挣值")
        second = embeddings.embed_query("什么是挣值")
        assert first == second and stub.count == 1                                      # 重复问题走缓存
        assert stub.requests[0][1]["input"] == ["Instruct: 检索\nQuery: 什么是挣值"]
        embeddings.embed_documents(["挣值是……"])
        assert stub.requests[1][1]["input"] == ["挣值是……"]                              # 文档不加前缀
        embeddings.embed_queries(["甲", "乙"])
        assert stub.requests[2][1]["input"] == ["Instruct: 检索\nQuery: 甲", "Instruct: 检索\nQuery: 乙"]
    with StubProviders() as stub:  # 默认不加前缀
        create_embeddings(_settings(stub)).embed_query("原样")
        assert stub.requests[0][1]["input"] == ["原样"]


def test_embeddings_errors_are_clear_and_never_leak_the_key():
    with StubProviders(status=401) as stub:
        embeddings = create_embeddings(_settings(stub))
        exc = _raises(ProviderError, lambda: embeddings.embed_documents(["x"]), "HTTP 401", "密钥", "embedding")
        assert KEY not in str(exc) and "***" in str(exc), "服务端回显的密钥必须被抹掉"
        assert stub.count == 1                                                          # 401 不重试
    with StubProviders(fail_first=2) as stub:  # 5xx 重试后成功
        embeddings = create_embeddings(_settings(stub, embedding_max_retries=2))
        assert len(embeddings.embed_documents(["x"])) == 1 and stub.count == 3
    with StubProviders(fail_first=99) as stub:
        embeddings = create_embeddings(_settings(stub, embedding_max_retries=1))
        exc = _raises(ProviderError, lambda: embeddings.embed_documents(["x"]), "HTTP 500", "重试 1 次")
        assert stub.count == 2 and KEY not in str(exc)
    with StubProviders(drop_one=True) as stub:
        _raises(ProviderError, lambda: create_embeddings(_settings(stub)).embed_documents(["a", "b"]), "数量")
    with StubProviders(plain_text=True) as stub:
        _raises(ProviderError, lambda: create_embeddings(_settings(stub)).embed_documents(["a"]), "不是 JSON")
    with StubProviders(delay=2) as stub:
        embeddings = create_embeddings(_settings(stub, embedding_timeout=0.3))
        exc = _raises(ProviderError, lambda: embeddings.embed_documents(["a"]), "超时")
        assert exc.timeout is True
    # 连接被拒绝（端口没人监听）：网络错误，同样不含密钥
    exc = _raises(ProviderError, lambda: create_embeddings(
        _settings(embedding_provider="openai", embedding_base_url="http://127.0.0.1:9/v1", embedding_api_key=KEY,
                  embedding_max_retries=0)).embed_documents(["a"]), "网络错误")
    assert KEY not in str(exc)


def test_embeddings_empty_input_and_no_key_header():
    with StubProviders() as stub:
        embeddings = create_embeddings(_settings(stub, embedding_api_key=None))
        assert embeddings.embed_documents([]) == [] and stub.count == 0
        embeddings.embed_documents(["x"])
        assert "Authorization" not in stub.requests[0][0], "没有密钥就不发 Authorization 头（本机 Ollama 之类）"


# ---------------------------------------------------------------------------
# rerank API
# ---------------------------------------------------------------------------

def test_rerank_api_request_shape_scores_and_cache():
    docs = ["无关内容", "挣值分析用于衡量绩效", "关键路径", "挣值"]
    with StubProviders() as stub:
        reranker = create_reranker(_settings(stub))
        ranked = reranker.rerank_with_scores("挣值分析", docs, top_k=2)
        headers, body, path = stub.requests[0]
        assert path == "/v1/rerank" and headers["Authorization"] == f"Bearer {KEY}"
        assert body == {"model": "stub-rerank", "query": "挣值分析", "documents": docs, "top_n": 2}
        assert [index for index, _ in ranked] == [1, 3] and ranked[0][1] == 1.0 and ranked[1][1] == 0.5  # 分数按 index 映射
        assert reranker.rerank_with_scores("挣值分析", docs, top_k=2) == ranked and stub.count == 1  # 缓存
        assert reranker.rerank("挣值分析", docs, top_k=2) == [1, 3] and stub.count == 1
        assert [i for i, _ in reranker.rerank_with_scores("挣值分析", docs, top_k=10)] == [1, 3, 0, 2] and stub.count == 2
        assert stub.requests[1][1]["top_n"] == 4                                      # top_n 不超过候选数
        assert reranker.rerank_with_scores("q", []) == [] and stub.count == 2         # 空候选不请求


def test_rerank_api_ignoring_top_n_is_sorted_and_truncated():
    with StubProviders(rerank_all=True) as stub:  # 服务端忽略 top_n，且按升序返回
        reranker = create_reranker(_settings(stub))
        ranked = reranker.rerank_with_scores("挣值分析", ["无", "挣值分析", "挣值"], top_k=2)
    assert [index for index, _ in ranked] == [1, 2], ranked


def test_rerank_api_errors():
    with StubProviders(status=403) as stub:
        reranker = create_reranker(_settings(stub))
        exc = _raises(ProviderError, lambda: reranker.rerank_with_scores("q", ["a"]), "HTTP 403", "rerank")
        assert KEY not in str(exc)
    with StubProviders(fail_first=99) as stub:
        _raises(ProviderError, lambda: create_reranker(_settings(stub, reranker_max_retries=1)).rerank("q", ["a"]), "HTTP 500")
        assert stub.count == 2

    class FixedClient:  # 直接喂畸形响应
        def __init__(self, data):
            self.data = data

        def post(self, path, payload):
            return self.data

    for data, needle in [({}, "results"), ({"results": [{"index": 5, "relevance_score": 0.5}]}, "无效"),
                         ({"results": [{"index": 0}]}, "relevance_score"), ({"results": "x"}, "results")]:
        _raises(ProviderError, lambda: ApiReranker(FixedClient(data), "m").rerank_with_scores("q", ["a", "b"]), needle)
    # 兼容只给 score 字段的实现
    ranked = ApiReranker(FixedClient({"results": [{"index": 1, "score": 0.9}, {"index": 0, "score": 0.2}]}), "m").rerank_with_scores("q", ["a", "b"])
    assert ranked == [(1, 0.9), (0, 0.2)]


# ---------------------------------------------------------------------------
# provider = none：不精排
# ---------------------------------------------------------------------------

class _Retriever:
    bm25_ready = True

    def __init__(self, count=4):
        self.count = count

    def catalog(self):
        return [{"name": "book.pdf", "chunk_count": self.count, "heading_chunks": 0, "has_headings": False}]

    def resolve_sources(self, names):
        return [f"/data/{name}" for name in names]

    def invoke(self, question, mode=None, sources=None):
        return [
            Document(page_content=f"片段文本{i}", metadata={
                "chunk_id": f"c{i}", "source": "/data/book.pdf", "page": i, "vec_rank": i + 1, "bm25_rank": None, "rrf_score": 0.03 - i * 0.001,
            })
            for i in range(self.count)
        ]


class _Chain:
    def __init__(self):
        self.calls = []

    def invoke(self, payload):
        self.calls.append(payload)
        return "【依据】\n- [片段1] “片段文本0”\n【回答】\n答案 [片段1]"


def _no_rerank_pipeline(count=4):
    chain = _Chain()
    return RAGPipeline(retriever=_Retriever(count), reranker=NoReranker(), answer_chain=chain, embeddings=None, device="cloud"), chain


def test_none_reranker_keeps_recall_order_with_null_scores_and_never_refuses_by_score():
    pipeline, chain = _no_rerank_pipeline()
    docs, candidates = retrieve_and_rerank(pipeline, "q", rerank_k=3)
    assert [d.metadata["chunk_id"] for d in docs] == ["c0", "c1", "c2"] and len(candidates) == 4   # 召回顺序取前 k
    assert all(d.metadata["rerank_score"] is None for d in docs)
    assert not should_refuse(docs, 0.99) and select_context_docs(docs, 0.99) == docs               # None 不触发门槛
    result = ask(pipeline, "q", rerank_k=3)                                                        # 默认阈值 0.7 / 0.95
    assert result.refused is False and result.top_score is None and len(result.docs) == 3 and len(chain.calls) == 1
    assert result.answer == "答案 [片段1]"
    # 空召回仍然拒答，且不调用 LLM
    empty, empty_chain = _no_rerank_pipeline(count=0)
    refused = ask(empty, "q")
    assert refused.refused is True and refused.answer == NO_ANSWER and not empty_chain.calls


def test_none_reranker_through_http_search_and_ask_and_health():
    settings = _settings(reranker_provider="none")
    original = api.collection_count
    api.collection_count = lambda *a, **k: 4
    logging.disable(logging.CRITICAL)
    try:
        pipeline, _ = _no_rerank_pipeline()
        app = api.create_app(pipeline_factory=lambda _s: pipeline, settings=settings)
        with TestClient(app) as client:
            search = client.post("/search", json={"query": "q", "top_k": 2}).json()
            assert search["refused"] is False and search["top_score"] is None, search
            assert [c["scores"]["rerank_score"] for c in search["chunks"]] == [None, None]
            answer = client.post("/ask", json={"query": "q"}).json()
            assert answer["refused"] is False and answer["top_score"] is None and answer["citations"], answer
            assert all(c["scores"]["rerank_score"] is None for c in answer["citations"])
            health = client.get("/health").json()
            assert health["reranker"] == {"provider": "none", "model": None}, health
            assert health["embedding"] == {"provider": "local", "model": "Qwen/Qwen3-Embedding-0.6B"}
            assert health["device"] == "cloud"                                                # 流水线里的设备
    finally:
        logging.disable(logging.NOTSET)
        api.collection_count = original


def test_health_reports_providers_and_cloud_device_without_pipeline():
    with StubProviders() as stub:
        settings = _settings(stub)
        original = api.collection_count
        api.collection_count = lambda *a, **k: None  # 索引不存在：不建流水线
        try:
            with TestClient(api.create_app(pipeline_factory=lambda _s: None, settings=settings)) as client:
                health = client.get("/health").json()
        finally:
            api.collection_count = original
    assert health["status"] == "not_ready" and health["device"] == "cloud"
    assert health["embedding"] == {"provider": "openai", "model": "stub-embed"}
    assert health["reranker"] == {"provider": "api", "model": "stub-rerank"}
    assert KEY not in json.dumps(health)


# ---------------------------------------------------------------------------
# 配置与工厂
# ---------------------------------------------------------------------------

def test_unknown_provider_is_a_clear_validation_error():
    for field, value in [("embedding_provider", "cohere"), ("reranker_provider", "magic"), ("embedding_provider", "")]:
        try:
            Settings(_env_file=None, **{field: value})
        except ValueError as exc:
            assert field in str(exc), exc
        else:
            raise AssertionError(f"{field}={value!r} 应被拒绝")
    assert Settings(_env_file=None, embedding_provider=" OpenAI ", reranker_provider="API").embedding_provider == "openai"


def test_defaults_are_local_and_unchanged():
    with clean_env():
        settings = Settings(_env_file=None)
    assert (settings.embedding_provider, settings.reranker_provider) == ("local", "local")
    assert settings.uses_local_models and settings.uses_local_embedding and settings.uses_local_reranker
    assert provider_problems(settings) == []
    assert settings.index_info_path == settings.meta_file.with_name(".index_info.json")
    assert index_info.current_signature(settings) == index_info.LEGACY_SIGNATURE


def test_endpoint_falls_back_to_llm_settings_without_leaking_to_other_hosts():
    llm = dict(llm_base_url="https://llm.example.com/v1", llm_api_key=LLM_KEY, llm_model="m", llm_timeout=33, llm_max_retries=5)
    settings = Settings(_env_file=None, embedding_provider="openai", reranker_provider="api", **llm)
    for endpoint in (settings.embedding_endpoint(), settings.reranker_endpoint()):          # 全部回退
        assert endpoint.base_url == "https://llm.example.com/v1" and endpoint.api_key.get_secret_value() == LLM_KEY
        assert (endpoint.timeout, endpoint.max_retries) == (33, 5)
        assert endpoint.base_url_from_llm and endpoint.api_key_from_llm
    # 显式指定了别家地址：不能把大模型的密钥发过去
    other = Settings(_env_file=None, embedding_provider="openai", embedding_base_url="https://other.example.com/v1", **llm)
    assert other.embedding_endpoint().api_key is None and other.embedding_endpoint().base_url == "https://other.example.com/v1"
    # 显式给了密钥就用它；地址与大模型同源（路径不同）时可沿用大模型密钥
    own = Settings(_env_file=None, embedding_provider="openai", embedding_base_url="https://other.example.com/v1", embedding_api_key=KEY,
                   embedding_timeout=7, embedding_max_retries=0, **llm)
    assert own.embedding_endpoint().api_key.get_secret_value() == KEY and (own.embedding_endpoint().timeout, own.embedding_endpoint().max_retries) == (7, 0)
    same = Settings(_env_file=None, embedding_provider="openai", embedding_base_url="https://LLM.example.com/embed-v2", **llm)
    assert same.embedding_endpoint().api_key.get_secret_value() == LLM_KEY and not same.embedding_endpoint().base_url_from_llm
    # 通过真实请求验证：回退的地址 + 密钥确实被用上
    with StubProviders() as stub:
        fallback = Settings(_env_file=None, embedding_provider="openai", llm_base_url=stub.base_url, llm_api_key=LLM_KEY, llm_model="m")
        create_embeddings(fallback).embed_documents(["x"])
        assert stub.requests[0][0]["Authorization"] == f"Bearer {LLM_KEY}"


def test_secrets_not_in_repr_str_or_client():
    settings = _settings(None, embedding_provider="openai", embedding_base_url="http://x/v1", embedding_api_key=KEY,
                         reranker_provider="api", reranker_base_url="http://x/v1", reranker_api_key=KEY, llm_api_key=LLM_KEY)
    blobs = [repr(settings), str(settings), repr(settings.embedding_endpoint()), repr(settings.reranker_endpoint()),
             repr(ApiClient(settings.embedding_endpoint(), "embedding")), str(settings.model_dump())]
    for blob in blobs:
        assert KEY not in blob and LLM_KEY not in blob, blob[:200]
    problems = _settings(None, embedding_provider="openai")
    assert KEY not in "".join(provider_problems(problems))


def test_missing_endpoint_gives_clear_errors():
    settings = _settings(None, embedding_provider="openai", reranker_provider="api")
    problems = provider_problems(settings)
    assert any("RAG_EMBEDDING_BASE_URL" in text for text in problems) and any("RAG_RERANKER_BASE_URL" in text for text in problems)
    _raises(ProviderConfigError, lambda: create_embeddings(settings), "RAG_EMBEDDING_BASE_URL")
    _raises(ProviderConfigError, lambda: create_reranker(settings), "RAG_RERANKER_BASE_URL")
    bad = _settings(None, embedding_provider="openai", embedding_base_url="api.example.com/v1")
    _raises(ProviderConfigError, lambda: create_embeddings(bad), "http")
    # none 不需要任何地址；一个提供方有问题不影响创建另一个
    assert isinstance(create_reranker(_settings(None, reranker_provider="none")), NoReranker)
    only_embedding_bad = _settings(None, embedding_provider="openai", reranker_provider="none")
    assert isinstance(create_reranker(only_embedding_bad), NoReranker)


def test_factory_local_providers_use_configured_names_without_loading_models():
    from ragbase.models import embeddings as embeddings_module
    from ragbase.models import reranker as reranker_module

    seen = []

    class FakeEmbeddings:
        def __init__(self, model_name=None, device=None):
            seen.append(("embedding", model_name, device))

    class FakeReranker:
        def __init__(self, model_name=None, device=None):
            seen.append(("reranker", model_name, device))

    original = (embeddings_module.Qwen3Embeddings, reranker_module.Qwen3Reranker)
    embeddings_module.Qwen3Embeddings, reranker_module.Qwen3Reranker = FakeEmbeddings, FakeReranker
    try:
        settings = _settings(None, embedding_model="my/embed", reranker_model="my/rerank", device="cpu")
        assert isinstance(create_embeddings(settings), FakeEmbeddings) and isinstance(create_reranker(settings), FakeReranker)
        assert isinstance(create_embeddings(settings, device="mps"), FakeEmbeddings)
    finally:
        embeddings_module.Qwen3Embeddings, reranker_module.Qwen3Reranker = original
    assert seen == [("embedding", "my/embed", "cpu"), ("reranker", "my/rerank", "cpu"), ("embedding", "my/embed", "mps")]


# ---------------------------------------------------------------------------
# models download / doctor
# ---------------------------------------------------------------------------

def test_models_download_only_covers_local_providers():
    all_cloud = _settings(None, embedding_provider="openai", reranker_provider="api")
    assert download.required_models(all_cloud) == [] and download.missing_models(all_cloud) == []
    mixed = _settings(None, embedding_provider="openai", reranker_provider="local", reranker_model="my/rerank")
    assert [(s.role, s.repo_id) for s in download.required_models(mixed)] == [("reranker", "my/rerank")]
    assert [s.role for s in download.required_models(_settings(None, reranker_provider="none"))] == ["embedding"]
    lines: list[str] = []
    assert download.ensure_models(all_cloud, download=lambda repo: (_ for _ in ()).throw(AssertionError("不应下载")), out=lines.append) == []
    assert sum("云端提供方，无需下载" in line for line in lines) == 2, lines
    code, out = _run_cli(_fresh_dir("dl"), ["models", "download"], RAG_EMBEDDING_PROVIDER="openai", RAG_RERANKER_PROVIDER="none")
    assert code in (0, None) and "无需下载" in out, out


def test_doctor_provider_checks_never_print_secrets_and_warn_on_min_score():
    with StubProviders() as stub:
        settings = _settings(stub, llm_api_key=LLM_KEY, llm_base_url=stub.base_url, llm_model="m")
        checks = doctor.check_providers(settings)
        text = "\n".join(f"{c.level} {c.text} {c.hint}" for c in checks)
        assert "stub-embed" in text and "stub-rerank" in text and "密钥已设置" in text
        assert KEY not in text and LLM_KEY not in text and stub.count == 0, "默认不调用接口"
        assert any(c.level == doctor.WARN and "calibrate_rerank_threshold" in c.hint for c in checks), text   # 0.7 仍是默认值
        tuned = doctor.check_providers(_settings(stub, min_score=0.4))
        assert not any("calibrate_rerank_threshold" in c.hint for c in tuned)
        # 密钥沿用大模型的
        fallback = Settings(_env_file=None, embedding_provider="openai", llm_base_url=stub.base_url, llm_api_key=LLM_KEY, llm_model="m")
        assert "沿用 RAG_LLM_API_KEY" in "\n".join(c.text for c in doctor.check_providers(fallback))
    default = doctor.check_providers(_settings(None))
    assert all(c.level == doctor.OK for c in default) and not any("calibrate" in c.hint for c in default)
    local_other = doctor.check_providers(_settings(None, reranker_model="my/rerank"))
    assert any("calibrate_rerank_threshold" in c.hint for c in local_other)
    # 缺地址 -> FAIL；无密钥 -> WARN；none -> WARN
    missing = doctor.check_providers(_settings(None, embedding_provider="openai"))
    assert any(c.level == doctor.FAIL and "RAG_EMBEDDING_BASE_URL" in c.hint for c in missing)
    nokey = doctor.check_providers(_settings(None, embedding_provider="openai", embedding_base_url="http://localhost:11434/v1"))
    assert any(c.level == doctor.WARN and "未设置密钥" in c.text for c in nokey)
    none = doctor.check_providers(_settings(None, reranker_provider="none"))
    assert any(c.level == doctor.WARN and "不会按相关度拒答" in c.text for c in none)
    # 全云端：跳过设备 / 模型缓存检查（不导入 torch、不查 HF 缓存）
    cloud = _settings(None, embedding_provider="openai", embedding_base_url="http://x/v1", reranker_provider="none")
    local_checks = doctor._local_model_checks(cloud)
    assert len(local_checks) == 1 and "跳过设备与模型缓存检查" in local_checks[0].text
    assert len(doctor._local_model_checks(_settings(None, embedding_provider="openai", embedding_base_url="http://x/v1"))) > 1  # reranker 仍是本地


def test_doctor_check_models_makes_one_call_per_cloud_provider():
    with StubProviders() as stub:
        settings = _settings(stub)
        assert doctor.check_providers(settings).__len__() and stub.count == 0
        checks = doctor.check_providers(settings, call=True)
        assert stub.paths() == ["/v1/embeddings", "/v1/rerank"], stub.paths()              # 各一次
        assert [c.level for c in checks[-2:]] == [doctor.OK, doctor.OK] and "向量维度 16" in checks[-2].text
    # 探测失败：错误里的密钥被抹掉
    def failing():
        raise RuntimeError(f"401 for {KEY}")

    with StubProviders() as stub:
        bad = doctor.check_providers(_settings(stub), call=True, embed_probe=failing, rerank_probe=lambda: None)
        assert stub.count == 0
        failed = [c for c in bad if c.level == doctor.FAIL]
        assert len(failed) == 1 and KEY not in failed[0].text and "***" in failed[0].text
    # 本地 / none 不发请求
    assert doctor.check_providers(_settings(None, reranker_provider="none"), call=True)[-1].level != doctor.FAIL
    parser = cli.build_parser()
    assert parser.parse_args(["doctor", "--check-models"]).check_models is True and parser.parse_args(["doctor"]).check_models is False


# ---------------------------------------------------------------------------
# 索引签名守卫
# ---------------------------------------------------------------------------

class FakeStore:
    name = "fake"
    location = "/fake"

    def __init__(self, count=5, dim=None):
        self._count, self._dim = count, dim

    def exists(self):
        return self._count > 0

    def count(self):
        return self._count

    def iter_all(self, batch_size=1000, with_vectors=False):
        if self._dim is not None:
            yield type("Rec", (), {"id": "a", "vector": [0.1] * self._dim})()


def _guard_settings(name, **kwargs) -> Settings:
    home = _fresh_dir(name)
    return Settings(_env_file=None, meta_file=home / ".ingest_meta.json", llm_model=None, llm_api_key=None, llm_base_url=None, **kwargs)


def test_guard_empty_index_has_no_constraint():
    settings = _guard_settings("g-empty", embedding_provider="openai", embedding_base_url="http://x/v1")
    for store in (FakeStore(count=0),):
        assert index_info.ensure_index_compatible(settings, store).status == "empty"
    assert not settings.index_info_path.exists()


def test_guard_legacy_index_default_config_is_written_silently_but_doctor_mode_is_read_only():
    settings = _guard_settings("g-legacy")
    assert index_info.check_index(settings, FakeStore(), write_legacy=False).status == "legacy_default"
    assert not settings.index_info_path.exists(), "只读检查不应写文件"
    assert index_info.ensure_index_compatible(settings, FakeStore()).status == "legacy_default"
    info = json.loads(settings.index_info_path.read_text(encoding="utf-8"))
    assert info["embedding"] == {"provider": "local", "model": "Qwen/Qwen3-Embedding-0.6B", "dimensions": None}
    assert index_info.ensure_index_compatible(settings, FakeStore()).status == "match"


def test_guard_legacy_index_with_non_default_config_is_refused():
    for kwargs in (dict(embedding_provider="openai", embedding_base_url="http://x/v1", embedding_model="bge-m3"),
                   dict(embedding_model="other/local-model")):
        settings = _guard_settings("g-legacy2", **kwargs)
        exc = _raises(IndexMismatchError, lambda: index_info.ensure_index_compatible(settings, FakeStore()),
                      "当前 embedding 配置与索引不一致", "ragbase ingest --full", "旧索引没有 embedding 记录")
        assert settings.embedding_model in str(exc) and "Qwen/Qwen3-Embedding-0.6B" in str(exc)
        assert not settings.index_info_path.exists(), "拒绝时不能改写签名"


def test_guard_signature_match_mismatch_and_rewrite():
    cloud = dict(embedding_provider="openai", embedding_base_url="http://x/v1", embedding_model="bge-m3", embedding_dimensions=512)
    settings = _guard_settings("g-sig", **cloud)
    store = FakeStore()
    IndexRecorder(settings, store).begin()                                                    # 入库登记
    assert index_info.ensure_index_compatible(settings, store).status == "match"
    # 地址 / 密钥 / 批大小不属于签名
    moved = Settings(_env_file=None, meta_file=settings.meta_file, **{**cloud, "embedding_base_url": "http://other/v1", "embedding_batch_size": 3})
    assert index_info.ensure_index_compatible(moved, store).status == "match"
    for changes in (dict(embedding_model="bge-large"), dict(embedding_dimensions=1024), dict(embedding_dimensions=None),
                    dict(embedding_provider="local"), dict(embedding_provider="local", embedding_model="Qwen/Qwen3-Embedding-0.6B")):
        changed = Settings(_env_file=None, meta_file=settings.meta_file, **{**cloud, **changes})
        _raises(IndexMismatchError, lambda: index_info.ensure_index_compatible(changed, store), "ragbase ingest --full", "bge-m3")
    # --full 重新登记后，新配置通过、旧配置反过来被拒
    new = Settings(_env_file=None, meta_file=settings.meta_file, **{**cloud, "embedding_model": "bge-large"})
    settings.index_info_path.unlink()
    IndexRecorder(new, store).begin()
    assert index_info.ensure_index_compatible(new, store).status == "match"
    _raises(IndexMismatchError, lambda: index_info.ensure_index_compatible(settings, store), "bge-large")
    # 签名文件损坏：拒绝并给出提示
    settings.index_info_path.write_text("{坏的", encoding="utf-8")
    _raises(IndexMismatchError, lambda: index_info.ensure_index_compatible(new, store), "索引信息文件", "无法读取")


def test_guard_checks_vector_length_against_existing_index():
    settings = _guard_settings("g-dim", embedding_provider="openai", embedding_base_url="http://x/v1")
    store = FakeStore(dim=8)
    recorder = IndexRecorder(settings, store)
    recorder.begin()
    embed = recorder.wrap(lambda texts: [[0.1] * 4 for _ in texts])
    _raises(IndexMismatchError, lambda: embed(["a"]), "向量长度 4", "已有向量的长度 8", "--full")
    good = IndexRecorder(settings, store).wrap(lambda texts: [[0.1] * 8 for _ in texts])
    assert len(good(["a", "b"])) == 2
    assert json.loads(settings.index_info_path.read_text(encoding="utf-8"))["dimension"] == 8     # 记录实际维度
    # 空索引：不比较，只记录
    fresh = _guard_settings("g-dim2", embedding_provider="openai", embedding_base_url="http://x/v1")
    fresh_recorder = IndexRecorder(fresh, FakeStore(count=0))
    fresh_recorder.begin()
    fresh_recorder.wrap(lambda texts: [[0.1] * 3 for _ in texts])(["a"])
    assert json.loads(fresh.index_info_path.read_text(encoding="utf-8"))["dimension"] == 3


def test_doctor_reports_index_signature():
    settings = _guard_settings("g-doctor", embedding_provider="openai", embedding_base_url="http://x/v1", embedding_model="bge-m3")
    checks = doctor.check_index_signature(settings, FakeStore())
    assert checks[0].level == doctor.FAIL and "bge-m3" in checks[0].text and "--full" in checks[0].hint
    assert not settings.index_info_path.exists()
    IndexRecorder(settings, FakeStore()).begin()
    assert doctor.check_index_signature(settings, FakeStore())[0].level == doctor.OK
    legacy = _guard_settings("g-doctor2")
    assert doctor.check_index_signature(legacy, FakeStore())[0].level == doctor.OK and not legacy.index_info_path.exists()
    legacy.index_info_path.write_text("garbage", encoding="utf-8")
    assert doctor.check_index_signature(legacy, FakeStore())[0].level == doctor.FAIL


# ---------------------------------------------------------------------------
# 命令行端到端（云端 embedding 桩 + 真实 Chroma 临时目录）
# ---------------------------------------------------------------------------

def _run_cli(root: Path, argv: list[str], **env) -> tuple[int, str]:
    """在隔离的环境变量里运行 ragbase.cli.main，返回 (退出码, 标准输出)。全云端配置会把 torch 等标记为不可导入，结束后还原。"""
    values = {"RAG_HOME": str(root), "RAG_DB_DIR": str(root / "chroma_db"), "RAG_DATA_DIR": str(root / "data"),
              "RAG_META_FILE": str(root / "meta.json"), **env}
    saved_modules = {name: sys.modules.get(name, "<absent>") for name in config.LOCAL_ML_MODULES}
    out, code = io.StringIO(), 0
    with clean_env(**values):
        try:
            with contextlib.redirect_stdout(out):
                try:
                    code = cli.main(argv)
                except SystemExit as exc:
                    code = exc.code
        finally:
            for name, module in saved_modules.items():
                if module == "<absent>":
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module
    return code, out.getvalue()


def _cloud_env(stub: StubProviders, model="stub-embed", **extra) -> dict:
    return {"RAG_EMBEDDING_PROVIDER": "openai", "RAG_EMBEDDING_BASE_URL": stub.base_url, "RAG_EMBEDDING_API_KEY": KEY,
            "RAG_EMBEDDING_MODEL": model, "RAG_RERANKER_PROVIDER": "none", "RAG_EMBEDDING_MAX_RETRIES": "0", **extra}


def _write_docs(root: Path):
    (root / "data").mkdir(exist_ok=True)
    for name, text in [("a.md", "挣值分析用于衡量项目绩效。" * 8), ("b.md", "关键路径决定项目最短工期。" * 8)]:
        (root / "data" / name).write_text(text, encoding="utf-8")


def test_cli_ingest_records_signature_and_refuses_after_embedding_change():
    root = _fresh_dir("cli-guard")
    _write_docs(root)
    with StubProviders() as stub:
        env = _cloud_env(stub)
        code, out = _run_cli(root, ["ingest"], **env)
        assert code == 0 and "完成" in out, out
        info_path = root / ".index_info.json"
        info = json.loads(info_path.read_text(encoding="utf-8"))
        assert info["embedding"] == {"provider": "openai", "model": "stub-embed", "dimensions": None} and info["dimension"] == 16
        assert stub.requests and all(h["Authorization"] == f"Bearer {KEY}" for h, _, _ in stub.requests)
        code, out = _run_cli(root, ["ingest"], **env)                                       # 同配置增量：没有变化
        assert code == 0 and "没有变化" in out, out

        calls = stub.count
        changed = _cloud_env(stub, model="other-embed")
        code, out = _run_cli(root, ["ingest"], **changed)                                   # 增量被拒
        assert code == 1 and "ragbase ingest --full" in out and "other-embed" in out and "stub-embed" in out, out
        code, out = _run_cli(root, ["serve"], **changed)                                    # serve 被拒（在加载任何东西之前）
        assert code == 1 and "ragbase ingest --full" in out, out
        code, out = _run_cli(root, ["ask"], RAG_LLM_BASE_URL=stub.base_url, RAG_LLM_API_KEY=LLM_KEY, RAG_LLM_MODEL="m", **changed)
        assert "ragbase ingest --full" in out and "加载" not in out, out                     # ask 在加载模型前拒绝
        code, out = _run_cli(root, ["chat"], RAG_LLM_BASE_URL=stub.base_url, RAG_LLM_API_KEY=LLM_KEY, RAG_LLM_MODEL="m", **changed)
        assert "ragbase ingest --full" in out, out
        assert stub.count == calls, "拒绝时不应调用任何接口"

        code, out = _run_cli(root, ["ingest", "--full"], **changed)                          # --full 重建并改写签名
        assert code == 0 and "完成" in out, out
        assert json.loads(info_path.read_text(encoding="utf-8"))["embedding"]["model"] == "other-embed"
        code, out = _run_cli(root, ["ingest"], **changed)
        assert code == 0 and "没有变化" in out, out
        code, out = _run_cli(root, ["ingest"], **env)                                       # 旧配置现在反过来被拒
        assert code == 1 and "ragbase ingest --full" in out, out


def test_cli_reports_incomplete_cloud_config_without_traceback():
    root = _fresh_dir("cli-badcfg")
    _write_docs(root)
    code, out = _run_cli(root, ["ingest"], RAG_EMBEDDING_PROVIDER="openai")
    assert code == 1 and "RAG_EMBEDDING_BASE_URL" in out and "Traceback" not in out, out
    code, out = _run_cli(root, ["ingest", "--full"], RAG_EMBEDDING_PROVIDER="openai")
    assert code == 1 and "RAG_EMBEDDING_BASE_URL" in out, out
    code, out = _run_cli(root, ["serve"], RAG_RERANKER_PROVIDER="api")
    assert code == 1 and "RAG_RERANKER_BASE_URL" in out, out


# ---------------------------------------------------------------------------
# 全云端：不依赖 torch（子进程里把 torch / transformers / sentence_transformers 标记为不可导入）
# ---------------------------------------------------------------------------

CLOUD_SCRIPT = textwrap.dedent(
    """
    import sys
    from ragbase import cli
    from ragbase.config import LOCAL_ML_MODULES, Settings

    LOCAL = LOCAL_ML_MODULES
    assert not any(name in sys.modules for name in LOCAL), "导入 cli 时不应加载 torch 等"
    assert cli.main(["ingest"]) == 0  # 全云端配置：命令行入口会把它们标记为不可导入（langchain 不再顺带加载）
    assert all(sys.modules.get(name, "absent") is None for name in LOCAL), {n: sys.modules.get(n, "absent") for n in LOCAL}

    from ragbase.retrieval.pipeline import build_pipeline, retrieve_and_rerank
    from ragbase.models.api_reranker import ApiReranker
    settings = Settings()
    pipeline, llm = build_pipeline(settings=settings, build_bm25=True, mode="hybrid")
    assert pipeline.device == "cloud" and llm is None
    assert isinstance(pipeline.reranker, ApiReranker)
    docs, candidates = retrieve_and_rerank(pipeline, "挣值分析是什么", rerank_k=2)
    assert docs and docs[0].metadata["source"].endswith("a.md"), [d.metadata["source"] for d in docs]
    assert docs[0].metadata["rerank_score"] is not None
    assert all(sys.modules.get(name, "absent") is None for name in LOCAL), "构建流水线 + 检索 + 精排期间没有导入 torch 等"
    pipeline.retriever.store.close()
    print("CLOUD-OK")
    """
)


def test_all_cloud_config_runs_without_torch():
    root = _fresh_dir("cloud-sub")
    _write_docs(root)
    with StubProviders() as stub:
        env = {key: value for key, value in os.environ.items() if not key.startswith(("RAG_", "ARK_"))}
        env.update({
            "RAG_HOME": str(root), "RAG_VECTOR_STORE": "chroma", "RAG_LOG_LEVEL": "WARNING",
            "RAG_EMBEDDING_PROVIDER": "openai", "RAG_EMBEDDING_BASE_URL": stub.base_url, "RAG_EMBEDDING_API_KEY": KEY,
            "RAG_EMBEDDING_MODEL": "stub-embed", "RAG_RERANKER_PROVIDER": "api", "RAG_RERANKER_BASE_URL": stub.base_url,
            "RAG_RERANKER_API_KEY": KEY, "RAG_RERANKER_MODEL": "stub-rerank", "CHROMA_TELEMETRY_DISABLED": "true",
        })
        completed = subprocess.run([sys.executable, "-c", CLOUD_SCRIPT], cwd=ROOT, env=env, capture_output=True, text=True,
                                   encoding="utf-8", errors="replace", timeout=300)
        assert completed.returncode == 0 and "CLOUD-OK" in completed.stdout, completed.stdout[-1500:] + completed.stderr[-2500:]
        assert "/v1/rerank" in stub.paths() and stub.paths().count("/v1/embeddings") >= 2
        assert KEY not in completed.stdout + completed.stderr


def test_block_local_ml_imports_only_blocks_missing_modules():
    saved = {name: sys.modules.get(name, "<absent>") for name in config.LOCAL_ML_MODULES}
    try:
        for name in config.LOCAL_ML_MODULES:
            sys.modules.pop(name, None)
        sys.modules["transformers"] = sys.modules  # 假装已导入
        blocked = config.block_local_ml_imports()
        assert blocked == ["torch", "sentence_transformers"] and sys.modules["torch"] is None and sys.modules["transformers"] is sys.modules
        try:
            import torch  # noqa: F401
        except ImportError:
            pass
        else:
            raise AssertionError("被标记的模块应无法导入")
    finally:
        for name, module in saved.items():
            if module == "<absent>":
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    started = time.perf_counter()
    for test in tests:
        test()
        print(f"通过：{test.__name__}")
    print(f"共 {len(tests)} 项测试通过（{time.perf_counter() - started:.1f}s）")
