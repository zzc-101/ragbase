"""题集 / 体检 / 报告 HTTP API 的测试：FastAPI TestClient + 假流水线（不加载模型、不访问网络、不调用大模型）。

数据库、数据目录都在临时目录里。运行：uv run python tests/test_api_diagnose.py
"""
import os
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from fastapi.testclient import TestClient
from langchain_core.documents import Document

from ragbase.api import app as api
from ragbase.config import Settings
from ragbase.retrieval.pipeline import RAGPipeline


class _Retriever:
    """返回固定候选；gate 不为 None 时每次检索都先等它放行（entered 表示已有检索进来）。"""

    bm25_ready = True
    store = None

    def __init__(self, gate: threading.Event | None = None):
        self.gate = gate
        self.entered = threading.Event()

    def catalog(self):  # DELETE /documents/{name} 在抢“忙”锁之前会先查书目
        return [{"name": "book.pdf", "chunk_count": 2, "heading_chunks": 0, "has_headings": False}]

    def resolve_sources(self, names):
        return [f"/d/{name}" for name in names]

    def invoke(self, question, mode=None, sources=None, timings=None):
        self.entered.set()
        if self.gate is not None:
            assert self.gate.wait(10), "测试没有放行检索"
        if timings is not None:
            timings["embed"] = 0.001
        return [Document(page_content=f"{question} 的候选片段{i}", metadata={"chunk_id": f"c{i}", "source": "/d/book.pdf"}) for i in range(2)]


class _Reranker:
    def rerank_with_scores(self, query, documents, top_k=5):
        return [(0, 0.9), (1, 0.5)][:top_k]


def _pipeline(gate=None):
    return RAGPipeline(retriever=_Retriever(gate), reranker=_Reranker(), answer_chain=None, embeddings=None, device="cpu")


@contextmanager
def _client(pipeline=None, count=100, **overrides):
    """每次一个全新的临时目录（含数据库文件）；index 计数用假值，lifespan 里的流水线工厂返回传入的假流水线。"""
    pipeline = pipeline or _pipeline()
    original, original_home = api.collection_count, os.environ.get("RAG_HOME")
    api.collection_count = lambda *args, **kwargs: count
    try:
        with tempfile.TemporaryDirectory(prefix="ragdemo-api-diagnose-") as tmp:
            root = Path(tmp)
            os.environ["RAG_HOME"] = tmp  # 首次打开数据库会导入 RAG_HOME/eval_questions.json：指向临时目录，别读到项目里的真实题集
            settings = Settings(
                _env_file=None, device="cpu", log_level="WARNING", vector_store="chroma",
                llm_api_key=None, llm_model=None, llm_base_url=None,
                data_dir=root / "data", db_dir=root / "chroma", meta_file=root / "meta.json",
                cache_dir=root / "cache", db_path=root / "ragbase.db", web_dist=root / "no-dist",
                **overrides,
            )
            app = api.create_app(pipeline_factory=lambda _settings: pipeline, settings=settings)
            with TestClient(app) as client:
                yield client, pipeline
    finally:
        api.collection_count = original
        if original_home is None:
            os.environ.pop("RAG_HOME", None)
        else:
            os.environ["RAG_HOME"] = original_home


def _add(client, *texts, reference="要点"):
    for text in texts:
        response = client.post("/eval-questions", json={"question": text, "reference": reference})
        assert response.status_code == 201, response.text


def _wait_job(client, job_id, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/diagnose/jobs/{job_id}").json()
        if job["status"] in ("succeeded", "failed"):
            return job
        time.sleep(0.02)
    raise AssertionError("体检任务超时未结束")


def test_diagnose_flow_report_history_and_delete():
    with _client() as (client, _):
        _add(client, "什么是关键路径", "什么是挣值")
        response = client.post("/diagnose", json={"mode": "bm25"})
        assert response.status_code == 202, response.text
        accepted = response.json()
        assert set(accepted) == {"job_id", "status", "total", "done", "report_id", "error", "started_at", "finished_at"}
        assert accepted["total"] == 2
        job = _wait_job(client, accepted["job_id"])
        assert job["status"] == "succeeded" and job["error"] is None, job
        assert (job["done"], job["total"]) == (2, 2) and job["finished_at"]

        items = client.get("/diagnose/reports").json()["items"]
        assert [item["id"] for item in items] == [job["report_id"]]
        assert items[0]["origin"] == "web" and items[0]["mode"] == "bm25" and items[0]["question_count"] == 2

        report = client.get(f"/diagnose/reports/{job['report_id']}").json()
        assert report["id"] == job["report_id"] and report["origin"] == "web"
        assert len(report["questions"]) == 2 and report["params"]["mode"] == "bm25"
        assert report["concurrent_chat_requests"] == 0 and len(report["question_ids"]) == 2

        assert client.delete(f"/diagnose/reports/{job['report_id']}").json() == {"deleted": True}
        assert client.get(f"/diagnose/reports/{job['report_id']}").status_code == 404
        assert client.delete(f"/diagnose/reports/{job['report_id']}").status_code == 404
        assert client.get("/diagnose/reports").json() == {"items": []}
        assert client.get("/diagnose/jobs/nope").status_code == 404


def test_import_validation_422_and_export_attachment():
    with _client() as (client, _):
        _add(client, "原有题目")
        before = client.get("/eval-questions").json()
        response = client.post(
            "/eval-questions/import",
            json={"items": [{"question": "好题"}, {"reference": "缺 question"}, {"question": 5}], "mode": "replace"},
        )
        assert response.status_code == 422, response.text
        body = response.json()
        assert body["error"] == "validation_error" and isinstance(body["detail"], str) and body["detail"]
        assert [(e["index"], e["field"]) for e in body["errors"]] == [(1, "question"), (2, "question")]
        assert all(e["message"] for e in body["errors"])
        assert client.get("/eval-questions").json() == before  # 整体没写入，replace 也没清空

        not_array = client.post("/eval-questions/import", json={"items": {"question": "x"}})  # 不是数组：整体错误，不是 pydantic 的 422
        assert not_array.status_code == 422
        assert not_array.json()["errors"][0]["index"] is None and "数组" in not_array.json()["errors"][0]["message"]
        assert client.get("/eval-questions").json() == before

        exported = client.get("/eval-questions/export")
        assert exported.status_code == 200
        assert exported.headers["content-disposition"] == 'attachment; filename="eval_questions.json"'
        data = exported.json()
        assert isinstance(data, list) and data[0]["question"] == "原有题目"


def test_ingest_rejected_while_diagnosing():
    gate = threading.Event()
    with _client(_pipeline(gate)) as (client, pipeline):
        _add(client, "什么是关键路径")
        job_id = client.post("/diagnose").json()["job_id"]
        assert pipeline.retriever.entered.wait(5)
        for response in (client.post("/ingest"), client.post("/diagnose"), client.delete("/documents/book.pdf")):
            assert response.status_code == 409, response.text
            assert response.json()["error"] == "conflict" and "体检" in response.json()["detail"]
        gate.set()
        assert _wait_job(client, job_id)["status"] == "succeeded"
        again = client.post("/diagnose")  # 结束后立刻可以再启动
        assert again.status_code == 202
        assert _wait_job(client, again.json()["job_id"])["status"] == "succeeded"


def test_question_snapshot_is_taken_at_start():
    gate = threading.Event()
    with _client(_pipeline(gate)) as (client, pipeline):
        _add(client, "题一", "题二", "题三")
        job_id = client.post("/diagnose").json()["job_id"]
        assert pipeline.retriever.entered.wait(5)
        first = client.get("/eval-questions").json()["items"][0]["id"]
        assert client.delete(f"/eval-questions/{first}").json() == {"deleted": True}  # 体检进行中删除一道题
        gate.set()
        job = _wait_job(client, job_id)
        assert job["status"] == "succeeded" and job["total"] == 3
        report = client.get(f"/diagnose/reports/{job['report_id']}").json()
        assert report["summary"]["questions"]["measured"] == 3 and len(report["questions"]) == 3
        assert client.get("/diagnose/reports").json()["items"][0]["question_count"] == 3


def test_diagnose_with_empty_question_set_is_400():
    with _client() as (client, _):
        response = client.post("/diagnose")
        assert response.status_code == 400, response.text
        assert response.json()["error"] == "bad_request" and "题集" in response.json()["detail"]


def test_auth_protects_api_but_diagnose_page_is_public():
    with _client(api_key="secret-key") as (client, _):
        for method, path in (("get", "/eval-questions"), ("post", "/diagnose"), ("get", "/diagnose/reports"),
                             ("get", "/eval-questions/export"), ("post", "/eval-questions/import")):
            response = getattr(client, method)(path)
            assert response.status_code == 401, (path, response.status_code)
        page = client.get("/diagnose")  # 前端路由：公开
        assert page.status_code == 200 and page.headers["content-type"].startswith("text/html")
        headers = {"Authorization": "Bearer secret-key"}
        assert client.get("/eval-questions", headers=headers).json() == {"items": []}
        assert client.post("/diagnose", headers=headers).status_code == 400  # 带密钥后才轮到业务判断（题集为空）


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"通过：{test.__name__}")
    print(f"共 {len(tests)} 项测试通过")
