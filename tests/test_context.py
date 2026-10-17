"""上下文过滤与“先摘录依据再回答”的纯单元测试（假 pipeline，不加载模型、不调用 LLM）。

运行：uv run python tests/test_context.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document

from rag_pipeline import (
    ANSWER_SYSTEM_PROMPT,
    CONTEXT_MIN_SCORE,
    NO_ANSWER,
    RAGPipeline,
    ask,
    format_docs,
    select_context_docs,
    split_evidence_answer,
)


def _docs(scores):
    return [
        Document(page_content=f"片段文本{i}", metadata={"chunk_id": f"c{i}", "rerank_score": score})
        for i, score in enumerate(scores)
    ]


def _ids(docs):
    return [doc.metadata["chunk_id"] for doc in docs]


def test_select_thresholds():
    docs = _docs([0.99, 0.8, 0.5, 0.1])
    assert _ids(select_context_docs(docs, 0.5)) == ["c0", "c1", "c2"]  # 边界含等于
    assert _ids(select_context_docs(docs, 0.6)) == ["c0", "c1"]
    assert _ids(select_context_docs(docs, 0.99)) == ["c0"]
    assert _ids(select_context_docs(docs, None)) == ["c0", "c1", "c2", "c3"]
    assert _ids(select_context_docs(docs, 0)) == ["c0", "c1", "c2", "c3"]
    assert select_context_docs([], 0.5) == []


def test_select_always_keeps_top1_and_order():
    docs = _docs([0.4, 0.9, 0.2])  # top1 自己低于阈值也保留
    assert _ids(select_context_docs(docs, 0.8)) == ["c0", "c1"]
    docs[1].metadata.pop("rerank_score")  # 缺分数视为满分
    assert _ids(select_context_docs(docs, 0.8)) == ["c0", "c1"]


def test_default_context_threshold_is_sane():
    assert 0 < CONTEXT_MIN_SCORE < 1


class _Retriever:
    def __init__(self, n):
        self.candidates = [
            Document(page_content=f"片段文本{i}", metadata={"chunk_id": f"c{i}"})
            for i in range(n)
        ]

    def invoke(self, question):
        return self.candidates


class _Reranker:
    def __init__(self, scores):
        self.scores = scores

    def rerank_with_scores(self, query, documents, top_k=5):
        return self.scores[:top_k]


class _Chain:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def invoke(self, payload):
        self.calls.append(payload)
        return self.output


def _pipeline(scores, output="【依据】\n- [片段1] “片段文本2”\n【回答】\n答案 [片段1]"):
    return RAGPipeline(
        retriever=_Retriever(4),
        reranker=_Reranker(scores),
        answer_chain=_Chain(output),
        embeddings=None,
    )


def test_ask_sends_only_filtered_docs():
    pipeline = _pipeline([(2, 0.95), (0, 0.8), (3, 0.3), (1, 0.05)])
    result = ask(pipeline, "q", min_score=0.7, context_min_score=0.5)
    context = pipeline.answer_chain.calls[0]["context"]
    assert context == "[片段1] 片段文本2\n\n[片段2] 片段文本0"
    assert "片段文本3" not in context and "片段文本1" not in context
    assert _ids(result.docs) == ["c2", "c0"]  # 与 [片段N] 一一对应
    assert len(result.reranked) == 4 and len(result.candidates) == 4
    assert not result.refused and result.top_score == 0.95
    assert result.evidence == "- [片段1] “片段文本2”"
    assert result.answer == "答案 [片段1]"


def test_ask_context_filter_disabled():
    pipeline = _pipeline([(0, 0.9), (1, 0.01)])
    result = ask(pipeline, "q", min_score=0.7, context_min_score=0)
    assert _ids(result.docs) == ["c0", "c1"]
    assert pipeline.answer_chain.calls[0]["context"].count("[片段") == 2


def test_ask_refused_returns_no_docs_for_llm():
    pipeline = _pipeline([(0, 0.1), (1, 0.05)])
    result = ask(pipeline, "q", min_score=0.7)
    assert result.refused and result.docs == [] and result.evidence == ""
    assert len(result.reranked) == 2 and pipeline.answer_chain.calls == []


def test_ask_llm_no_answer_and_unformatted():
    result = ask(_pipeline([(0, 0.9)], output=NO_ANSWER), "q")
    assert (result.evidence, result.answer) == ("", NO_ANSWER) and not result.refused
    result = ask(_pipeline([(0, 0.9)], output="直接回答 [片段1]"), "q")
    assert (result.evidence, result.answer) == ("", "直接回答 [片段1]")


def test_split_exact_format():
    text = "【依据】\n- [片段1] “原文一”\n- [片段2] “原文二”\n【回答】\n结论 [片段1][片段2]\n第二行"
    evidence, answer = split_evidence_answer(text)
    assert evidence == "- [片段1] “原文一”\n- [片段2] “原文二”"
    assert answer == "结论 [片段1][片段2]\n第二行"


def test_split_variants():
    cases = [
        "[依据]\n- a\n[回答]\n答",                    # 半角方括号
        "（依据）\n- a\n（回答）\n答",                # 全角圆括号
        "依据：\n- a\n\n\n回答：答",                  # 无括号 + 冒号 + 多余空行
        "**【依据】**\n- a\n\n**【回答】**\n答\n",     # markdown 加粗
        "  【依据】：\n\n- a\n\n【回答】：\n\n答  \n\n", # 标题后带冒号
        "### 依据:\n- a\n### 回答:\n答",
    ]
    for text in cases:
        assert split_evidence_answer(text) == ("- a", "答"), text


def test_split_no_answer_alone():
    assert split_evidence_answer(NO_ANSWER) == ("", NO_ANSWER)
    assert split_evidence_answer(f"  {NO_ANSWER}\n") == ("", NO_ANSWER)
    assert split_evidence_answer(f"“{NO_ANSWER}”") == ("", NO_ANSWER)


def test_split_unformatted_text():
    assert split_evidence_answer("直接回答 [片段1]") == ("", "直接回答 [片段1]")
    assert split_evidence_answer("回答该问题需要综合多个片段。") == ("", "回答该问题需要综合多个片段。")
    assert split_evidence_answer("") == ("", "")
    assert split_evidence_answer(None) == ("", "")
    # 只有依据没有回答：不丢内容，整体当回答
    only_evidence = "【依据】\n- [片段1] “原文”"
    assert split_evidence_answer(only_evidence) == ("", only_evidence)
    # 只有回答标题
    assert split_evidence_answer("【回答】\n答案") == ("", "答案")


def test_prompt_has_format_and_rules():
    for needle in ("【依据】", "【回答】", "[片段N]", "逐字", NO_ANSWER, "{context}"):
        assert needle in ANSWER_SYSTEM_PROMPT, needle


def test_format_docs_prefixes_heading_when_present():
    docs = [
        Document(page_content="正文甲", metadata={"heading_path": "第11章 项目成本管理 > 11.3 挣值分析"}),
        Document(page_content="正文乙", metadata={}),  # 旧版片段：没有章节
        Document(page_content="正文丙", metadata={"heading_path": ""}),
    ]
    assert format_docs(docs) == (
        "[片段1]（第11章 项目成本管理 > 11.3 挣值分析）正文甲\n\n[片段2] 正文乙\n\n[片段3] 正文丙"
    )
    assert format_docs([]) == ""


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"通过：{test.__name__}")
    print(f"共 {len(tests)} 项测试通过")
