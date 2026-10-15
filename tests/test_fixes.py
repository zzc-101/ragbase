"""评测裁判解析、增量入库规划、文档发现、空召回的纯单元测试（不加载模型）。

运行：uv run python tests/test_fixes.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from document_parser import list_documents
from evaluate import parse_answer_verdict, parse_retrieval_verdict
from ingest_incremental import plan_changes
from query import source_label
from rag_pipeline import NO_ANSWER, RAGPipeline, ask, retrieve_and_rerank


def test_answer_verdict():
    assert parse_answer_verdict("正确") == "正确"
    assert parse_answer_verdict("  “正确”。 ") == "正确"
    assert parse_answer_verdict("不正确") == "错误"
    assert parse_answer_verdict("错误") == "错误"
    assert parse_answer_verdict("部分正确") == "部分正确"
    assert parse_answer_verdict("部分") == "部分正确"
    assert parse_answer_verdict("") == "错误"
    assert parse_answer_verdict("无法判断") == "错误"
    # 多行：以第一个能识别标签的行为准
    assert parse_answer_verdict("\n判断：不正确\n理由：虽然部分内容正确") == "错误"
    assert parse_answer_verdict("分析如下\n\n部分正确\n要点不全") == "部分正确"


def test_retrieval_verdict():
    assert parse_retrieval_verdict("充分") == "充分"
    assert parse_retrieval_verdict("不充分") == "不足"
    assert parse_retrieval_verdict("不足") == "不足"
    assert parse_retrieval_verdict("部分") == "部分"
    assert parse_retrieval_verdict("部分充分") == "部分"
    assert parse_retrieval_verdict("") == "不足"
    assert parse_retrieval_verdict("评估结果：\n不充分，信息不全") == "不足"


def _meta(**items):
    return {fid: {"name": f"{fid}.pdf", "fp": fp} for fid, fp in items.items()}


def test_plan_changes_index_missing():
    old = _meta(a="1", b="2")
    cur = _meta(a="1", b="2")
    for count in (None, 0):
        to_add, deleted, unchanged, reason = plan_changes(old, cur, count)
        assert sorted(to_add) == ["a", "b"] and deleted == [] and unchanged == 0
        assert reason == "index_missing"
    # 旧元数据里已删除的文件在重建时不再处理
    to_add, deleted, _, reason = plan_changes(_meta(a="1", z="9"), cur, None)
    assert sorted(to_add) == ["a", "b"] and deleted == [] and reason == "index_missing"


def test_plan_changes_normal():
    old = _meta(a="1", b="2", c="3")
    cur = _meta(a="1", b="22", d="4")
    to_add, deleted, unchanged, reason = plan_changes(old, cur, 100)
    assert sorted(to_add) == ["b", "d"], to_add
    assert deleted == ["c"] and unchanged == 1 and reason is None
    to_add, deleted, unchanged, reason = plan_changes(old, old, 100)
    assert to_add == [] and deleted == [] and unchanged == 3 and reason is None


def test_plan_changes_legacy_and_first_run():
    cur = _meta(a="1", b="2")
    to_add, deleted, unchanged, reason = plan_changes({}, cur, 50)
    assert sorted(to_add) == ["a", "b"] and deleted == [] and reason == "legacy"
    # 全新安装：无元数据、无集合 -> 普通新增
    to_add, deleted, unchanged, reason = plan_changes({}, cur, None)
    assert sorted(to_add) == ["a", "b"] and reason is None
    to_add, _, _, reason = plan_changes({}, cur, 0)
    assert sorted(to_add) == ["a", "b"] and reason is None


def test_list_documents():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name in ("a.pdf", "B.PDF", "c.Docx", "d.md", "e.markdown", "f.txt",
                     "old.doc", "OLD2.DOC", "README.txt", "readme.MD", "x.png"):
            (root / name).write_text("x", encoding="utf-8")
        (root / "sub").mkdir()
        (root / "sub" / "nested.pdf").write_text("x", encoding="utf-8")
        docs, skipped = list_documents(root)
        assert [p.name for p in docs] == sorted(
            ["a.pdf", "B.PDF", "c.Docx", "d.md", "e.markdown", "f.txt"]
        ), docs
        assert [p.name for p in skipped] == ["OLD2.DOC", "old.doc"], skipped


def test_source_label():
    assert source_label({"source": "/x/a.pdf", "page": 0, "vec_rank": 2}) == "a.pdf:第1页[V2]"
    assert source_label({"source": "/x/a.md", "bm25_rank": 1, "vec_rank": 3}) == "a.md[V3+B1]"
    assert source_label({}) == "?[?]"


class _Boom:
    def __getattr__(self, name):
        raise AssertionError(f"不应被调用：{name}")


class _EmptyRetriever:
    def invoke(self, question):
        return []


def _empty_pipeline():
    return RAGPipeline(
        retriever=_EmptyRetriever(),
        reranker=_Boom(),
        answer_chain=_Boom(),
        embeddings=None,
    )


def test_empty_recall_skips_rerank_and_llm():
    pipeline = _empty_pipeline()
    assert retrieve_and_rerank(pipeline, "问题") == ([], [])
    result = ask(pipeline, "问题")
    assert (result.answer, result.docs, result.reranked, result.candidates) == (NO_ANSWER, [], [], [])
    assert result.refused and result.top_score is None
    assert NO_ANSWER == "我没有在文档中找到相关内容。"


def test_empty_results_metrics_ok():
    from evaluation_metrics import lexical_retrieval_metrics, mean_scores

    metrics = lexical_retrieval_metrics([], "要点一；要点二")
    assert metrics == {"point_recall": 0.0, "context_precision": 0.0, "mrr": 0.0}
    results = [{"lexical": {"retrieved_top20": metrics}}]
    assert mean_scores(results, ("lexical", "retrieved_top20"))["mrr"] == 0.0


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"通过：{test.__name__}")
    print(f"共 {len(tests)} 项测试通过")
