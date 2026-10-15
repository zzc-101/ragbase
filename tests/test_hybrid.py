"""hybrid_retriever 的纯单元测试（不加载模型、不访问 Chroma）。

运行：uv run python tests/test_hybrid.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hybrid_retriever import rrf_fuse, tokenize


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


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"通过：{test.__name__}")
    print(f"共 {len(tests)} 项测试通过")
