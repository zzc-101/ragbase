"""ragbase.retrieval.hybrid 的纯单元测试（不加载模型、不访问 Chroma）。

运行：uv run python tests/test_hybrid.py
"""
import tempfile
from pathlib import Path

from ragbase.retrieval import hybrid as hybrid_retriever
from ragbase.retrieval.hybrid import HybridRetriever, rrf_fuse, tokenize


def test_rrf_worked_example():
    # vector [A,B,C]、bm25 [B,D,E]：B 两路都命中，应排第一；A 次之，D 在 C/E 之前
    fused = rrf_fuse([["A", "B", "C"], ["B", "D", "E"]], k=60)
    order = [doc_id for doc_id, _ in fused]
    scores = dict(fused)
    assert order[:3] == ["B", "A", "D"], order
    assert set(order[3:]) == {"C", "E"}
    assert abs(scores["B"] - (1 / 62 + 1 / 61)) < 1e-12
    assert abs(scores["C"] - scores["E"]) < 1e-12  # C=向量第3，E=BM25第3


def test_rrf_weights_and_empty():
    assert rrf_fuse([[], []]) == []
    fused = rrf_fuse([["A"], ["B"]], weights=[1.0, 2.0])
    assert [doc_id for doc_id, _ in fused] == ["B", "A"]


def test_tokenize_keeps_acronyms_and_numbers():
    tokens = tokenize("TOGAF、ITIL 和 ISO 9001 是什么？")
    assert {"togaf", "itil", "iso", "9001"} <= set(tokens), tokens


def test_tokenize_merges_spaced_letters():
    assert "togaf" in tokenize("采用 T O G A F 架构框架")


def test_tokenize_drops_punctuation_and_stopwords():
    tokens = tokenize("挣值分析，（关键路径）！ A. B")
    assert not any(not token.strip() for token in tokens)
    assert not set(tokens) & set("，（）！. "), tokens
    assert "a" not in tokens and "b" not in tokens  # 单个字母丢弃
    assert "挣值" in tokens and "关键路径"[:2] in tokens
    assert "的" not in tokenize("项目的范围")


def test_tokenize_empty_query():
    assert tokenize("？！，。 ") == []


class _Collection:
    """极简 Chroma 假对象：get 返回全部片段，query 固定返回 d3 > d1 > d2 的向量排名。"""

    def __init__(self):
        self.data = {
            "ids": ["d1", "d2", "d3"],
            "documents": ["关键路径法计算总浮动时间", "挣值分析与进度偏差", "红烧肉的做法"],
            "metadatas": [{"source": "a.pdf"}, {"source": "b.pdf"}, {"source": "c.pdf"}],
        }

    def get(self, include=None):
        return self.data

    def query(self, query_embeddings, n_results, include=None):
        order = [2, 0, 1][:n_results]
        return {
            "ids": [[self.data["ids"][i] for i in order]],
            "documents": [[self.data["documents"][i] for i in order]],
            "metadatas": [[self.data["metadatas"][i] for i in order]],
        }


class _Embeddings:
    def embed_query(self, text):
        return [0.0]


def _ids(docs):
    return [doc.metadata["chunk_id"] for doc in docs]


def test_invoke_mode_override_per_call():
    with tempfile.TemporaryDirectory() as tmp:  # 别动真实的 .bm25_cache
        retriever = HybridRetriever(_Collection(), _Embeddings(), mode="vector", build_bm25=True, cache_dir=Path(tmp))
    assert retriever.mode == "vector" and retriever.bm25_ready
    assert _ids(retriever.invoke("关键路径法")) == ["d3", "d1", "d2"]  # 默认模式 vector
    bm25_docs = retriever.invoke("关键路径法", mode="bm25")
    assert _ids(bm25_docs)[0] == "d1" and bm25_docs[0].metadata["bm25_rank"] == 1
    assert bm25_docs[0].metadata["vec_rank"] is None
    hybrid_docs = retriever.invoke("关键路径法", mode="hybrid")
    assert hybrid_docs[0].metadata["vec_rank"] is not None and hybrid_docs[0].metadata["bm25_rank"] is not None
    assert retriever.mode == "vector"  # 按次覆盖，不改默认模式


def test_invoke_bm25_without_index_raises_clear_error():
    retriever = HybridRetriever(_Collection(), _Embeddings(), mode="vector")
    assert not retriever.bm25_ready and _ids(retriever.invoke("q")) == ["d3", "d1", "d2"]
    for mode in ("bm25", "hybrid"):
        try:
            retriever.invoke("q", mode=mode)
        except RuntimeError as exc:
            assert "BM25" in str(exc) and mode in str(exc)
        else:
            raise AssertionError("应抛出 RuntimeError")
    try:
        retriever.invoke("q", mode="nope")
    except ValueError:
        pass
    else:
        raise AssertionError("非法 mode 应抛出 ValueError")


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"通过：{test.__name__}")
    print(f"共 {len(tests)} 项测试通过")
