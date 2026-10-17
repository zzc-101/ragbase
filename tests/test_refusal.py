"""低相关度拒答的纯单元测试（假 retriever / reranker / answer_chain，不加载模型、不调用 LLM）。

运行：uv run python tests/test_refusal.py
"""
from langchain_core.documents import Document

from ragbase.retrieval.labels import source_label
from ragbase.models.reranker import _sigmoid
from ragbase.retrieval.pipeline import (
    ANSWER_SYSTEM_PROMPT,
    NO_ANSWER,
    RERANK_MIN_SCORE,
    RAGPipeline,
    ask,
    retrieve_and_rerank,
)


class _Retriever:
    def __init__(self):
        self.candidates = [
            Document(page_content=f"片段文本{i}", metadata={"chunk_id": f"c{i}"})
            for i in range(3)
        ]

    def invoke(self, question):
        return self.candidates


class _Reranker:
    """固定返回 (原始索引, 分数)，模拟精排结果。"""

    def __init__(self, scores):
        self.scores = scores

    def rerank_with_scores(self, query, documents, top_k=5):
        return self.scores[:top_k]


class _Chain:
    def __init__(self):
        self.calls = []

    def invoke(self, payload):
        self.calls.append(payload)
        return "模型回答 [片段1]"


def _pipeline(scores):
    return RAGPipeline(
        retriever=_Retriever(),
        reranker=_Reranker(scores),
        answer_chain=_Chain(),
        embeddings=None,
    )


def test_low_score_refuses_without_llm():
    pipeline = _pipeline([(2, 0.03), (0, 0.01)])
    result = ask(pipeline, "红烧肉怎么做", min_score=0.3)
    assert result.answer == NO_ANSWER and result.refused
    assert pipeline.answer_chain.calls == []
    assert result.docs == []  # 没有片段交给 LLM
    assert len(result.reranked) == 2 and len(result.candidates) == 3  # 仍返回片段，便于调试
    assert result.reranked[0].metadata["chunk_id"] == "c2"
    assert result.top_score == 0.03


def test_high_score_calls_llm():
    pipeline = _pipeline([(1, 0.93), (0, 0.4)])
    result = ask(pipeline, "什么是软件工程", min_score=0.3, context_min_score=0)
    assert result.answer == "模型回答 [片段1]" and not result.refused
    assert len(pipeline.answer_chain.calls) == 1
    assert "[片段1] 片段文本1" in pipeline.answer_chain.calls[0]["context"]
    assert [doc.metadata["chunk_id"] for doc in result.docs] == ["c1", "c0"]


def test_threshold_boundary_is_inclusive_for_answering():
    pipeline = _pipeline([(0, 0.3)])
    assert ask(pipeline, "q", min_score=0.3).answer != NO_ANSWER
    pipeline = _pipeline([(0, 0.2999)])
    assert ask(pipeline, "q", min_score=0.3).answer == NO_ANSWER


def test_min_score_none_or_zero_disables_gating():
    for min_score in (None, 0):
        pipeline = _pipeline([(0, 0.0001)])
        result = ask(pipeline, "q", min_score=min_score)
        assert result.answer == "模型回答 [片段1]"
        assert len(pipeline.answer_chain.calls) == 1


def test_default_threshold_is_applied():
    pipeline = _pipeline([(0, 0.0)])
    assert ask(pipeline, "q").answer == NO_ANSWER  # 默认 RERANK_MIN_SCORE > 0
    assert 0 < RERANK_MIN_SCORE < 1


def test_rerank_score_attached_without_mutating_candidates():
    pipeline = _pipeline([(2, 0.9), (0, 0.5)])
    docs, candidates = retrieve_and_rerank(pipeline, "q", rerank_k=2)
    assert [doc.metadata["rerank_score"] for doc in docs] == [0.9, 0.5]
    assert all("rerank_score" not in doc.metadata for doc in candidates)
    assert docs[0] is not candidates[2]
    # 再来一次不同的分数，不应受上次结果影响
    pipeline.reranker.scores = [(2, 0.1)]
    docs2, _ = retrieve_and_rerank(pipeline, "q", rerank_k=1)
    assert docs2[0].metadata["rerank_score"] == 0.1
    assert docs[0].metadata["rerank_score"] == 0.9


def test_prompt_mentions_no_answer_and_citation():
    assert NO_ANSWER in ANSWER_SYSTEM_PROMPT
    assert "[片段N]" in ANSWER_SYSTEM_PROMPT
    assert "{context}" in ANSWER_SYSTEM_PROMPT


def test_sigmoid():
    assert abs(_sigmoid(0.0) - 0.5) < 1e-12
    assert _sigmoid(10.0) > 0.9999 and _sigmoid(-10.0) < 0.0001
    assert _sigmoid(-1000.0) == 0.0 and _sigmoid(1000.0) == 1.0  # 不溢出


def test_source_label_with_score():
    meta = {"source": "/x/a.pdf", "page": 11, "vec_rank": 2, "bm25_rank": 1, "rerank_score": 0.931}
    assert source_label(meta) == "a.pdf:第12页[V2+B1|0.93]"


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"通过：{test.__name__}")
    print(f"共 {len(tests)} 项测试通过")
