"""ragbase diagnose 的纯单元测试（假 retriever / reranker / embeddings，不加载模型、不读真实索引、不访问网络）。

运行：uv run python tests/test_diagnose.py
"""
import contextlib
import copy
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

from langchain_core.documents import Document

from ragbase import diagnostics
from ragbase.config import get_settings
from ragbase.db import Database, open_database
from ragbase.diagnostics import (
    CANDIDATE_RECALL_LOW,
    CLOUD_LATENCY_MS,
    CONTEXT_RECALL_DROP,
    MIN_QUESTIONS_FOR_QUALITY_RULES,
    REFUSAL_RATE_HIGH,
    RERANK_RECALL_DROP,
    RERANK_SHARE_PCT,
    STAGES,
    build_report,
    chunk_length_stats,
    collect_params,
    dedupe_questions,
    digest_questions,
    load_questions,
    make_recommendations,
    measure_question,
    reference_of,
    render_report,
    run_diagnosis,
    save_report,
    summarize,
)
from ragbase.diagnostics_reports import ReportStore
from ragbase.evaluation.questions import QuestionStore
from ragbase.retrieval.hybrid import HybridRetriever
from ragbase.retrieval.pipeline import RAGPipeline, retrieve_and_rerank
from ragbase.vectorstores.base import Hit, Record


# ---------------------------------------------------------------- 假对象

class _Collection:
    """极简假向量库（同 test_hybrid）：query 固定返回 d3 > d1 > d2。"""

    IDS = ["d1", "d2", "d3"]
    TEXTS = ["关键路径法计算总浮动时间", "挣值分析与进度偏差", "红烧肉的做法"]
    METADATAS = [{"source": "a.pdf"}, {"source": "b.pdf"}, {"source": "c.pdf"}]

    def iter_all(self, batch_size=1000, with_vectors=False):
        for doc_id, text, metadata in zip(self.IDS, self.TEXTS, self.METADATAS):
            yield Record(id=doc_id, text=text, metadata=dict(metadata))

    def query(self, embedding, k, sources=None):
        order = [2, 0, 1][:k]
        return [Hit(id=self.IDS[i], text=self.TEXTS[i], metadata=dict(self.METADATAS[i]), score=0.0) for i in order]


class _Embeddings:
    def embed_query(self, text):
        return [0.0]


def _retriever(mode):
    return HybridRetriever(_Collection(), _Embeddings(), mode=mode, per_route_k=3, final_k=3,
                           cache_dir=Path(tempfile.mkdtemp()))


class _PlainRetriever:
    """invoke(self, question) 不接受 timings：timings=None 时必须仍可用。"""

    def __init__(self, texts=("片段文本0", "片段文本1", "片段文本2")):
        self.candidates = [Document(page_content=text, metadata={"chunk_id": f"c{i}"}) for i, text in enumerate(texts)]
        self.calls = 0

    def invoke(self, question):
        self.calls += 1
        return self.candidates


class _TimedRetriever(_PlainRetriever):
    def invoke(self, question, timings=None):
        self.calls += 1
        if timings is not None:
            timings["embed"] = 0.002
            timings["vector_search"] = 0.003
            timings["fuse"] = 0.001
        return self.candidates


class _Reranker:
    def __init__(self, scores):
        self.scores = scores

    def rerank_with_scores(self, query, documents, top_k=5):
        return self.scores[:top_k]


def _pipeline(retriever, scores=((1, 0.9), (0, 0.5)), device="cpu"):
    return RAGPipeline(retriever=retriever, reranker=_Reranker(list(scores)), answer_chain=None,
                       embeddings=None, device=device)


def _settings(**overrides):
    values = dict(
        per_route_k=30, retrieve_k=20, rerank_k=5, min_score=0.7, context_min_score=0.95,
        embedding_provider="local", embedding_model="emb-model",
        reranker_provider="local", reranker_model="rerank-model",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


# ---------------------------------------------------------------- timings 钩子

def test_hybrid_timings_keys_and_non_negative():
    for mode, expected in {
        "hybrid": {"embed", "vector_search", "bm25", "fuse"},
        "vector": {"embed", "vector_search", "fuse"},
        "bm25": {"bm25", "fuse"},
    }.items():
        timings = {}
        docs = _retriever("hybrid").invoke("关键路径法", mode=mode, timings=timings)
        assert docs, mode
        assert set(timings) == expected, (mode, timings)  # 没跑的阶段不写键
        assert all(isinstance(value, float) and value >= 0 for value in timings.values()), timings


def test_hybrid_results_identical_with_and_without_timings():
    retriever = _retriever("hybrid")
    plain = retriever.invoke("关键路径法")
    timed = retriever.invoke("关键路径法", timings={})
    assert [d.metadata["chunk_id"] for d in plain] == [d.metadata["chunk_id"] for d in timed]
    assert [d.metadata["rrf_score"] for d in plain] == [d.metadata["rrf_score"] for d in timed]


def test_retrieve_and_rerank_writes_timings():
    pipeline = _pipeline(_TimedRetriever())
    timings = {}
    reranked, candidates = retrieve_and_rerank(pipeline, "q", 5, timings=timings)
    assert len(candidates) == 3 and [d.metadata["chunk_id"] for d in reranked] == ["c1", "c0"]
    assert set(timings) == {"embed", "vector_search", "fuse", "rerank"}
    assert all(value >= 0 for value in timings.values())


def test_retrieve_and_rerank_without_timings_does_not_pass_argument():
    retriever = _PlainRetriever()  # invoke(self, question)：多传参数会抛 TypeError
    reranked, candidates = retrieve_and_rerank(_pipeline(retriever), "q", 5)
    assert retriever.calls == 1 and len(candidates) == 3 and len(reranked) == 2


def test_empty_recall_has_no_rerank_timing():
    timings = {}
    reranked, candidates = retrieve_and_rerank(_pipeline(_TimedRetriever(texts=())), "q", 5, timings=timings)
    assert reranked == [] and candidates == []
    assert "rerank" not in timings


# ---------------------------------------------------------------- 题集

def test_dedupe_questions_keeps_first_and_counts():
    questions = [
        {"id": 1, "question": "什么是软件工程？"},
        {"id": 2, "question": "什么是软件架构？"},
        {"id": 3, "question": " 什么是软件工程？ "},  # 仅首尾空白不同也算重复
        {"id": 4, "question": "什么是软件工程？"},
    ]
    unique, skipped = dedupe_questions(questions)
    assert [q["id"] for q in unique] == [1, 2] and skipped == 2


def test_reference_of_prefers_reference_and_handles_missing():
    assert reference_of({"reference": "完整参考答案", "answer_keyword": "关键词"}) == "完整参考答案"
    assert reference_of({"reference": "关键词要点"}) == "关键词要点"
    assert reference_of({"answer_keyword": "题库里只有旧字段名：导入时已转成 reference，这里不再读它"}) is None
    assert reference_of({}) is None
    assert reference_of({"reference": ""}) is None
    assert reference_of({"reference": None}) is None
    assert reference_of({"reference": "a，b"}) is None  # 拆不出 >= 2 字符的要点


def _store_with(questions: list[dict]) -> QuestionStore:
    store = QuestionStore(Database(":memory:"))
    store.import_items(questions, "replace")
    return store


def test_load_questions_empty_store_exits_with_message():
    try:
        load_questions(_store_with([]))
    except SystemExit as exc:
        assert "错误" in str(exc.code) and "体检页" in str(exc.code) and "ragbase questions import" in str(exc.code)
    else:
        raise AssertionError("题集为空应以 SystemExit 退出")


def test_load_questions_limit_and_dedupe():
    data = [{"id": i, "question": q, "reference": "要点一，要点二"} for i, q in enumerate(["甲", "乙", "甲", "丙"], 1)]
    store = _store_with(data)
    questions, skipped = load_questions(store)
    assert [q["question"] for q in questions] == ["甲", "乙", "丙"] and skipped == 1
    assert all(q["reference"] == "要点一，要点二" for q in questions)
    questions, skipped = load_questions(store, 3)  # 先取前 3 题，再去重
    assert [q["question"] for q in questions] == ["甲", "乙"] and skipped == 1


def test_digest_questions_ignores_order_and_tracks_content():
    base = [{"id": 1, "question": "甲", "reference": "要点甲"}, {"id": 2, "question": "乙", "reference": None}]
    digest = digest_questions(base)
    assert len(digest) == 32 and digest == digest_questions(list(reversed(base)))  # 与顺序无关
    assert digest != digest_questions([base[0], {**base[1], "question": "乙乙"}])  # 与内容有关
    assert digest != digest_questions([base[0], {**base[1], "reference": "新参考答案"}])
    assert digest != digest_questions([base[0], {**base[1], "id": 3}])
    assert digest != digest_questions(base[:1])
    assert digest_questions([{**q, "difficulty": "简单"} for q in base]) == digest  # 其它字段不参与


# ---------------------------------------------------------------- 逐题测量与汇总

def _measure(question, scores=((1, 0.9), (0, 0.5)), retriever=None, **kwargs):
    pipeline = _pipeline(retriever or _TimedRetriever(texts=("关键路径法计算总浮动时间", "挣值分析与进度偏差", "红烧肉的做法")), scores)
    kwargs.setdefault("min_score", 0.3)
    kwargs.setdefault("context_min_score", 0)
    return measure_question(pipeline, question, 5, **kwargs)


def test_measure_question_record_shape():
    record = _measure({"id": 7, "question": "关键路径法", "reference": "关键路径法计算总浮动时间，挣值分析"})
    assert record["id"] == 7 and record["has_reference"] and not record["refused"]
    assert set(record["timings_ms"]) == {"embed", "vector_search", "fuse", "rerank", "total"}
    assert all(value >= 0 for value in record["timings_ms"].values())
    assert record["candidate_count"] == 3 and record["reranked_count"] == 2 and record["sent_count"] == 2
    assert record["top1_rerank_score"] == 0.9
    assert set(record["lexical"]) == {"candidates", "reranked", "llm_context"}
    assert set(record["lexical"]["candidates"]) == {"point_recall", "context_precision", "mrr"}
    assert record["lexical"]["candidates"]["point_recall"] == 1.0


def test_measure_question_without_reference_has_no_lexical():
    record = _measure({"id": 1, "question": "没有参考答案的题"})
    assert not record["has_reference"] and record["lexical"] is None
    assert record["timings_ms"]["total"] >= 0 and record["candidate_count"] == 3


def test_measure_question_refused_sends_nothing():
    record = _measure({"id": 1, "question": "q", "reference": "关键路径法计算总浮动时间"},
                      scores=((2, 0.05),), min_score=0.3)
    assert record["refused"] and record["sent_count"] == 0
    assert record["lexical"]["llm_context"] == {"point_recall": 0.0, "context_precision": 0.0, "mrr": 0.0}
    assert record["lexical"]["reranked"]["point_recall"] == 0.0  # 精排后只剩无关片段


def test_measure_question_context_filter_matches_eval():
    # 与 eval --retrieval-only 一致：top1 总保留，其余低于 context_min_score 的被过滤
    record = _measure({"id": 1, "question": "q", "reference": "关键路径法计算总浮动时间，挣值分析"},
                      scores=((0, 0.99), (1, 0.5)), context_min_score=0.95)
    assert record["reranked_count"] == 2 and record["sent_count"] == 1


def _record(total=100.0, rerank=60.0, embed=20.0, candidates=10, refused=False, top1=0.9, lexical=None, sent=3, **extra):
    timings = {"embed": embed, "vector_search": 5.0, "fuse": 1.0, "total": total}
    if rerank is not None:
        timings["rerank"] = rerank
    group = lambda recall: {"point_recall": recall, "context_precision": recall, "mrr": recall}  # noqa: E731
    if lexical is True:
        lexical = {"candidates": group(1.0), "reranked": group(0.8), "llm_context": group(0.6)}
    return {
        "id": 1, "question": "q", "has_reference": lexical is not None, "timings_ms": timings,
        "candidate_count": candidates, "reranked_count": 5, "sent_count": sent,
        "top1_rerank_score": top1, "refused": refused, "lexical": lexical, **extra,
    }


def test_summarize_speed_statistics():
    records = [
        _record(total=100, rerank=60, embed=10, candidates=10),
        _record(total=200, rerank=120, embed=30, candidates=20),
        _record(total=300, rerank=180, embed=20, candidates=30),
    ]
    speed = summarize(records)["speed"]
    assert speed["total"]["mean"] == 200 and speed["total"]["median"] == 200 and speed["total"]["max"] == 300
    rerank = speed["stages"]["rerank"]
    assert (rerank["mean"], rerank["median"], rerank["max"]) == (120, 120, 180)
    assert rerank["share_pct"] == 60.0  # 360 / 600
    assert speed["stages"]["embed"]["share_pct"] == 10.0  # 60 / 600
    assert speed["stages"]["embed"]["median"] == 20
    assert "bm25" not in speed["stages"]  # 没跑的阶段不出现
    assert speed["rerank_ms_per_candidate"] == 6.0  # 360 ms / 60 条
    assert speed["avg_candidates"] == 20


def test_summarize_stage_missing_in_some_questions():
    records = [_record(rerank=None, candidates=0), _record(total=100, rerank=50)]
    speed = summarize(records)["speed"]
    assert speed["stages"]["rerank"]["count"] == 1 and speed["stages"]["rerank"]["mean"] == 50
    assert speed["stages"]["rerank"]["share_pct"] == 25.0  # 50 / (100 + 100)
    assert speed["rerank_ms_per_candidate"] == 5.0


def test_summarize_quality_only_counts_questions_with_reference():
    records = [_record(lexical=True), _record(lexical=None), _record(lexical=True, refused=True, top1=0.1)]
    summary = summarize(records, duplicates_skipped=2)
    assert summary["questions"] == {"measured": 3, "with_reference": 2, "duplicates_skipped": 2}
    quality = summary["quality"]
    assert quality["candidates"]["point_recall"] == 1.0 and quality["reranked"]["point_recall"] == 0.8
    assert quality["llm_context"]["mrr"] == 0.6
    assert quality["refused"] == 1 and abs(quality["refusal_rate"] - 1 / 3) < 1e-3
    assert quality["top1_score"]["min"] == 0.1 and quality["top1_score"]["max"] == 0.9 and quality["top1_score"]["count"] == 3
    assert quality["avg_sent_docs"] == 3
    # 过滤损失只看没拒答的题：只有第一题
    assert quality["context_filter"]["questions"] == 1 and abs(quality["context_filter"]["gap"] - 0.2) < 1e-9


def test_summarize_without_reference_or_scores():
    summary = summarize([_record(top1=None), _record(top1=None)])
    quality = summary["quality"]
    assert quality["candidates"] is None and quality["context_filter"] is None
    assert quality["top1_score"] is None
    assert summary["questions"]["with_reference"] == 0


def test_chunk_length_stats():
    assert chunk_length_stats([]) is None
    stats = chunk_length_stats(["a" * 10, "b" * 20, "c" * 60])
    assert stats == {"count": 3, "mean": 30.0, "median": 20.0, "max": 60}


def test_collect_params_local_and_cloud_reranker():
    local = SimpleNamespace(reranker=SimpleNamespace(_batch_size=8), device="mps")
    params = collect_params(_settings(), local, "hybrid", 3.456, ["x" * 4])
    assert params["reranker_batch_size"] == 8 and params["device"] == "mps"
    assert params["chunks"]["count"] == 1 and params["model_load_seconds"] == 3.46
    assert params["default_chunk_size"] == 500
    cloud = SimpleNamespace(reranker=object(), device="cloud")
    params = collect_params(_settings(embedding_provider="openai", reranker_provider="api"), cloud, "vector", 1.0, [])
    assert params["reranker_batch_size"] is None and params["device"] == "cloud" and params["chunks"] is None
    none = collect_params(_settings(reranker_provider="none"), cloud, "hybrid", 1.0, [])
    assert none["reranker_model"] is None and none["reranker_batch_size"] is None


# ---------------------------------------------------------------- 建议规则

def _base_data():
    """不会触发任何规则的基线数据；各规则的测试在它上面改一个值。"""
    params = {
        "mode": "hybrid", "per_route_k": 30, "retrieve_k": 20, "rerank_k": 5, "min_score": 0.7, "context_min_score": 0.95,
        "embedding_provider": "local", "reranker_provider": "local", "device": "mps",
    }
    stat = lambda mean, share: {"mean": mean, "median": mean, "max": mean, "min": mean, "count": 10, "share_pct": share}  # noqa: E731
    summary = {
        "questions": {"measured": 10, "with_reference": 10, "duplicates_skipped": 0},
        "quality": {
            "candidates": {"point_recall": 0.9, "context_precision": 0.5, "mrr": 0.8},
            "reranked": {"point_recall": 0.85, "context_precision": 0.6, "mrr": 0.8},
            "llm_context": {"point_recall": 0.8, "context_precision": 0.7, "mrr": 0.8},
            "context_filter": {"questions": 10, "reranked_point_recall": 0.85, "llm_point_recall": 0.8, "gap": 0.05},
            "refused": 1, "refusal_rate": 0.1, "top1_score": None, "avg_sent_docs": 3.0,
        },
        "speed": {
            "total": stat(100.0, 100.0),
            "stages": {"embed": stat(20.0, 20.0), "rerank": stat(50.0, 50.0)},
            "avg_candidates": 20.0, "rerank_ms_per_candidate": 2.5,
        },
    }
    return {"params": params, "summary": summary}


def _ids(data):
    return [item["id"] for item in make_recommendations(data)]


def _with(data, path, value):
    data = copy.deepcopy(data)
    target = data
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return data


def test_baseline_triggers_nothing():
    assert make_recommendations(_base_data()) == []


def test_rule_a_rerank_share_boundary():
    path = ["summary", "speed", "stages", "rerank", "share_pct"]
    assert "rerank_slow" not in _ids(_with(_base_data(), path, RERANK_SHARE_PCT - 0.1))
    items = make_recommendations(_with(_base_data(), path, RERANK_SHARE_PCT))  # 恰为阈值：触发（>=）
    item = next(item for item in items if item["id"] == "rerank_slow")
    assert "RAG_RETRIEVE_K" in item["try"] and "RAG_RERANKER_MODEL" in item["try"] and "RAG_RERANKER_PROVIDER=api" in item["try"]
    assert "60%" in item["phenomenon"] and "2.5" in item["phenomenon"]
    assert set(item) == {"id", "title", "phenomenon", "try", "cost"}
    # 已经是云端精排：不再建议“改用云端精排”
    cloud = _with(_with(_base_data(), path, 90.0), ["params", "reranker_provider"], "api")
    cloud_item = next(item for item in make_recommendations(cloud) if item["id"] == "rerank_slow")
    assert "RAG_RERANKER_PROVIDER=api" not in cloud_item["try"]


def test_rule_b_cpu_device():
    path = ["params", "device"]
    assert "cpu_device" not in _ids(_with(_base_data(), path, "mps"))
    assert "cpu_device" in _ids(_with(_base_data(), path, "cpu"))
    cloud = _with(_with(_with(_base_data(), path, "cpu"), ["params", "embedding_provider"], "openai"), ["params", "reranker_provider"], "api")
    assert "cpu_device" not in _ids(cloud)  # 没有本地模型就不适用
    only_reranker = _with(_with(_base_data(), path, "cpu"), ["params", "embedding_provider"], "openai")
    assert "cpu_device" in _ids(only_reranker)  # 只要有一个本地模型


def test_rule_c_candidate_recall_boundary_and_mode():
    path = ["summary", "quality", "candidates", "point_recall"]
    assert "low_candidate_recall" not in _ids(_with(_base_data(), path, CANDIDATE_RECALL_LOW))  # 恰为阈值：不触发（<）
    # 召回 0.5 同时让规则 d 的差值变大，这里只看规则 c
    data = _with(_with(_base_data(), path, 0.59), ["summary", "quality", "reranked", "point_recall"], 0.59)
    item = next(item for item in make_recommendations(data) if item["id"] == "low_candidate_recall")
    assert "改用 hybrid" not in item["try"]  # 当前已是 hybrid：不建议改 hybrid
    assert "RAG_PER_ROUTE_K" in item["try"] and "RAG_RETRIEVE_K" in item["try"] and "bm25_userdict.txt" in item["try"]
    vector = _with(data, ["params", "mode"], "vector")
    item = next(item for item in make_recommendations(vector) if item["id"] == "low_candidate_recall")
    assert "改用 hybrid" in item["try"]


def test_rule_d_rerank_drop_boundary():
    reranked = ["summary", "quality", "reranked", "point_recall"]
    at_threshold = _with(_base_data(), reranked, round(0.9 - RERANK_RECALL_DROP, 4))  # 差值恰为 0.15：不触发
    assert "rerank_drops_relevant" not in _ids(at_threshold)
    over = _with(_base_data(), reranked, round(0.9 - RERANK_RECALL_DROP - 0.01, 4))
    item = next(item for item in make_recommendations(over) if item["id"] == "rerank_drops_relevant")
    assert "RAG_RERANK_K" in item["try"] and "RAG_RERANKER_MODEL" in item["try"]
    none = _with(over, ["params", "reranker_provider"], "none")
    item = next(item for item in make_recommendations(none) if item["id"] == "rerank_drops_relevant")
    assert "RAG_RERANKER_MODEL" not in item["try"]  # 没有精排模型可换


def test_rule_e_context_filter_boundary():
    path = ["summary", "quality", "context_filter", "gap"]
    assert "context_filter_strict" not in _ids(_with(_base_data(), path, CONTEXT_RECALL_DROP))  # 恰为阈值：不触发
    item = next(item for item in make_recommendations(_with(_base_data(), path, 0.11)) if item["id"] == "context_filter_strict")
    assert "RAG_CONTEXT_MIN_SCORE" in item["try"] and "0.11" in item["phenomenon"]
    few = _with(_with(_base_data(), path, 0.3), ["summary", "quality", "context_filter", "questions"], MIN_QUESTIONS_FOR_QUALITY_RULES - 1)
    assert "context_filter_strict" not in _ids(few)
    assert "context_filter_strict" not in _ids(_with(_base_data(), ["summary", "quality", "context_filter"], None))


def test_rule_f_refusal_rate_boundary():
    rate = ["summary", "quality", "refusal_rate"]
    assert "high_refusal" not in _ids(_with(_base_data(), rate, REFUSAL_RATE_HIGH))  # 恰为阈值：不触发
    data = _with(_with(_base_data(), rate, 0.3), ["summary", "quality", "refused"], 3)
    item = next(item for item in make_recommendations(data) if item["id"] == "high_refusal")
    assert "scripts/calibrate_rerank_threshold.py" in item["try"] and "RAG_MIN_SCORE" in item["try"]
    assert "3/10" in item["phenomenon"]
    assert "high_refusal" not in _ids(_with(data, ["params", "reranker_provider"], "none"))
    assert "high_refusal" not in _ids(_with(data, ["summary", "questions", "measured"], MIN_QUESTIONS_FOR_QUALITY_RULES - 1))


def test_calibrate_script_exists():
    assert (Path(__file__).resolve().parent.parent / "scripts" / "calibrate_rerank_threshold.py").is_file()


def test_rule_g_no_reranker():
    assert "no_reranker" not in _ids(_base_data())
    data = _with(_base_data(), ["params", "reranker_provider"], "none")
    item = next(item for item in make_recommendations(data) if item["id"] == "no_reranker")
    assert "拒答" in item["try"]


def test_rule_h_cloud_latency_boundary():
    cloud = _with(_base_data(), ["params", "embedding_provider"], "openai")
    path = ["summary", "speed", "stages", "embed", "mean"]
    assert "cloud_latency" not in _ids(_with(cloud, path, CLOUD_LATENCY_MS))  # 恰为阈值：不触发（>）
    data = _with(cloud, path, CLOUD_LATENCY_MS + 1)
    item = next(item for item in make_recommendations(data) if item["id"] == "cloud_latency")
    assert "embedding" in item["phenomenon"] and "精排" not in item["phenomenon"]
    # 本地 embedding 再慢也不属于“网络延迟”
    assert "cloud_latency" not in _ids(_with(_base_data(), path, 5000.0))
    # 云端精排
    api = _with(_with(_base_data(), ["params", "reranker_provider"], "api"), ["summary", "speed", "stages", "rerank", "mean"], 2000.0)
    item = next(item for item in make_recommendations(api) if item["id"] == "cloud_latency")
    assert "精排" in item["phenomenon"]


def test_quality_rules_need_enough_questions_with_reference():
    data = _with(_base_data(), ["summary", "quality", "candidates", "point_recall"], 0.1)
    assert "low_candidate_recall" in _ids(data)
    few = _with(data, ["summary", "questions", "with_reference"], MIN_QUESTIONS_FOR_QUALITY_RULES - 1)
    assert "low_candidate_recall" not in _ids(few)
    none = _with(_with(data, ["summary", "quality", "candidates"], None), ["summary", "quality", "reranked"], None)
    assert make_recommendations(none) == []  # 没有召回指标也不应报错


# ---------------------------------------------------------------- 报告结构、排版与文件

def _report(recommend=True):
    records = [
        _record(total=100, rerank=80, lexical=True, candidates=20),
        _record(total=120, rerank=90, lexical=True, candidates=20),
        _record(total=110, rerank=85, lexical=None, candidates=20),
    ]
    params = collect_params(
        _settings(), SimpleNamespace(reranker=SimpleNamespace(_batch_size=8), device="cpu"), "hybrid", 2.0, ["x" * 100, "y" * 300]
    )
    return build_report(params, records, duplicates_skipped=1, generated_at="2026-01-01T00:00:00")


def test_build_report_structure_and_json_roundtrip():
    report = _report()
    assert set(report) == {"generated_at", "params", "warmup_question", "questions", "summary", "recommendations",
                           "question_ids", "question_digest", "concurrent_chat_requests"}
    assert report["question_ids"] == [1, 1, 1] and report["question_digest"] is None and report["concurrent_chat_requests"] == 0
    assert len(report["questions"]) == 3 and report["summary"]["questions"]["duplicates_skipped"] == 1
    ids = [item["id"] for item in report["recommendations"]]
    assert "rerank_slow" in ids and "cpu_device" in ids  # 精排占比约 77%，设备是 cpu
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "diagnose_report.json"
        save_report(report, path)
        assert not (Path(tmp) / "diagnose_report.tmp").exists()  # 原子写入后不留临时文件
        text = path.read_text(encoding="utf-8")
        assert "预热用的占位问题" in text  # ensure_ascii=False：中文原样写入
        assert json.loads(text) == json.loads(json.dumps(report, ensure_ascii=False))


def test_render_report_contains_all_sections():
    text = render_report(_report())
    for expected in ("【当前参数】", "【质量】", "【速度】", "【建议】", "per_route_k", "retrieve_k", "rerank_k", "min_score",
                     "context_min_score", "embedding", "reranker 批大小", "计算设备", "片段长度", "chunk_size=500", "模型加载耗时",
                     "参与统计的题数", "point_recall", "拒答率", "Top1 精排得分", "精排每条候选平均", "RAG_RETRIEVE_K",
                     "现象：", "可以尝试：", "可能的代价：", "调整参数后重新运行"):
        assert expected in text, expected
    assert "重复题已跳过" in text


def test_render_report_notes_concurrent_chat_requests():
    report = _report()
    assert "对话请求" not in render_report(report)  # 0 次：不提示
    report["concurrent_chat_requests"] = 7
    assert "体检期间有 7 次对话请求，耗时可能偏高" in render_report(report)
    old = _report()
    del old["concurrent_chat_requests"]  # 旧报告没有这个字段
    assert "对话请求" not in render_report(old)


def test_render_report_cloud_marks_network_and_no_recommendation_message():
    report = _report()
    report["params"].update(embedding_provider="openai", reranker_provider="api", reranker_batch_size=None, device="cloud")
    report["recommendations"] = []
    text = render_report(report)
    assert text.count("含网络往返") >= 2 and "不适用" in text and "未发现明显问题" in text
    none = _report()
    none["params"].update(reranker_provider="none", reranker_model=None)
    none["summary"]["quality"]["top1_score"] = None
    assert "none（不精排）" in render_report(none) and "无（未启用精排" in render_report(none)


def test_render_report_without_reference_questions():
    summary = summarize([_record(lexical=None)])
    report = {"params": _report()["params"], "summary": summary, "recommendations": []}
    assert "没有带参考答案的题" in render_report(report)


# ---------------------------------------------------------------- 命令行与模块依赖

def test_cli_registers_diagnose_with_options():
    from ragbase import cli

    args = cli.build_parser().parse_args(["diagnose", "--limit", "3", "--retrieval-mode", "vector", "--json", "out.json"])
    assert args.command == "diagnose" and args.limit == 3 and args.retrieval_mode == "vector" and args.json == "out.json"
    assert args.func is cli._cmd_diagnose
    args = cli.build_parser().parse_args(["diagnose"])
    assert args.limit is None and args.json is None and args.retrieval_mode in ("vector", "bm25", "hybrid")


# ---------------------------------------------------------------- run_diagnosis（体检核心）

class _ModeRetriever(_TimedRetriever):
    """invoke 接受 mode（与 HybridRetriever 一致）；记录每次调用收到的 mode。"""

    def __init__(self, texts=("关键路径法计算总浮动时间", "挣值分析与进度偏差", "红烧肉的做法")):
        super().__init__(texts)
        self.modes: list = []

    def invoke(self, question, mode=None, timings=None):
        self.modes.append(mode)
        return super().invoke(question, timings=timings)


_QUESTIONS = [
    {"id": 3, "question": "关键路径法怎么算", "reference": "关键路径法计算总浮动时间，挣值分析"},
    {"id": 1, "question": "没有参考答案的题", "reference": None},
    {"id": 2, "question": "挣值分析是什么", "reference": "挣值分析与进度偏差，成本偏差"},
]


def _run_diagnosis(questions=_QUESTIONS, **kwargs):
    retriever = _ModeRetriever()
    pipeline = _pipeline(retriever, device="cpu")
    report = run_diagnosis(pipeline, _settings(), questions, "bm25", 1.5, 2, **kwargs)
    return report, retriever


def test_run_diagnosis_report_and_progress():
    calls = []
    report, retriever = _run_diagnosis(on_progress=lambda done, total: calls.append((done, total)))
    assert calls == [(1, 3), (2, 3), (3, 3)]  # 每完成一题调用一次
    assert retriever.modes == ["bm25"] * 4  # 预热 + 3 题，都按指定的 mode 召回
    assert set(report) == {"generated_at", "params", "warmup_question", "questions", "summary", "recommendations",
                           "question_ids", "question_digest", "concurrent_chat_requests"}
    assert [r["id"] for r in report["questions"]] == [3, 1, 2]  # 预热不计入记录
    assert report["question_ids"] == [3, 1, 2] and report["question_digest"] == digest_questions(_QUESTIONS)
    assert report["params"]["mode"] == "bm25" and report["params"]["model_load_seconds"] == 1.5
    assert report["summary"]["questions"] == {"measured": 3, "with_reference": 2, "duplicates_skipped": 2}
    assert report["concurrent_chat_requests"] == 0  # 没传 chat_requests：记 0
    run_diagnosis(_pipeline(_ModeRetriever()), _settings(), [], "hybrid", 0.0, 0)  # 不传 on_progress 也行


def test_run_diagnosis_prints_nothing():
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        _run_diagnosis(on_progress=lambda done, total: None)
    assert buffer.getvalue() == ""


def test_run_diagnosis_concurrent_chat_requests_is_a_difference():
    counter = {"n": 100}  # 服务启动以来累计的对话请求数：开始时已经是 100

    def chat_requests():
        return counter["n"]

    def on_progress(done, total):
        counter["n"] += 2  # 体检期间每题之间插进来 2 次对话请求

    report, _ = _run_diagnosis(on_progress=on_progress, chat_requests=chat_requests)
    assert report["concurrent_chat_requests"] == 6
    assert "体检期间有 6 次对话请求" in render_report(report)


def test_run_diagnosis_digest_independent_of_question_order():
    first, _ = _run_diagnosis(_QUESTIONS)
    second, _ = _run_diagnosis(list(reversed(_QUESTIONS)))
    assert first["question_digest"] == second["question_digest"]
    changed = [{**_QUESTIONS[0], "question": "换了题目文字"}, *_QUESTIONS[1:]]
    assert _run_diagnosis(changed)[0]["question_digest"] != first["question_digest"]


@contextlib.contextmanager
def _env(**values):
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


def _run_cli_diagnose(home: Path, *argv) -> tuple[int, str]:
    """用假流水线跑 `ragbase diagnose`（不加载模型、不读真实索引），返回 (退出码, 标准输出)。"""
    from ragbase import cli
    from ragbase.ingest import common

    saved = (diagnostics._build_pipeline_or_exit, common.collection_count)
    diagnostics._build_pipeline_or_exit = lambda settings, mode: _pipeline(_ModeRetriever(), device="cpu")
    common.collection_count = lambda settings: 10
    buffer, code = io.StringIO(), 0
    try:
        with _env(RAG_HOME=str(home), RAG_DB_PATH=str(home / "cli-test.db")), contextlib.redirect_stdout(buffer):
            try:
                args = cli.build_parser().parse_args(["diagnose", *argv])
                args.func(args)
            except SystemExit as exc:
                code = exc.code if isinstance(exc.code, int) else 1
                buffer.write(str(exc.code) if not isinstance(exc.code, int) else "")
    finally:
        diagnostics._build_pipeline_or_exit, common.collection_count = saved
    return code, buffer.getvalue()


def test_cli_diagnose_saves_report_and_json_is_optional():
    with tempfile.TemporaryDirectory() as tmp:
        home = Path(tmp).resolve()
        with _env(RAG_HOME=str(home), RAG_DB_PATH=str(home / "cli-test.db")):
            db = open_database()
            QuestionStore(db).import_items(_QUESTIONS, "replace")
            db.close()

        code, out = _run_cli_diagnose(home)
        assert code == 0 and "RAG 体检报告" in out and "报告 id：1（可在 Web 体检页查看）" in out, out
        assert not (home / "diagnose_report.json").exists()  # 不再默认写文件

        with _env(RAG_HOME=str(home), RAG_DB_PATH=str(home / "cli-test.db")):
            db = open_database()
            stored = ReportStore(db).get(1)
            db.close()
        assert stored["origin"] == "cli" and stored["question_ids"] == [1, 2, 3]  # 题集按 id 排序读出

        export = home / "export.json"
        code, out = _run_cli_diagnose(home, "--json", str(export), "--limit", "2")
        assert code == 0 and "报告 id：2" in out and "已导出到" in out, out
        data = json.loads(export.read_text(encoding="utf-8"))
        assert data["summary"]["questions"]["measured"] == 2 and data["concurrent_chat_requests"] == 0
        assert not (home / "diagnose_report.json").exists()


def test_cli_diagnose_empty_question_set_and_database_error():
    with tempfile.TemporaryDirectory() as tmp:
        home = Path(tmp).resolve()
        code, out = _run_cli_diagnose(home)  # 临时主目录里没有 eval_questions.json，题集为空
        assert code != 0 and "题集是空的" in out and "ragbase questions import" in out, out
        blocker = home / "blocker"
        blocker.write_text("x", encoding="utf-8")
        from ragbase import cli

        with _env(RAG_HOME=str(home), RAG_DB_PATH=str(blocker / "sub" / "x.db")):
            args = cli.build_parser().parse_args(["diagnose"])
            try:
                args.func(args)
            except SystemExit as exc:
                assert "数据库" in str(exc.code) and "错误" in str(exc.code), exc.code
            else:
                raise AssertionError("数据库打不开应以 SystemExit 退出")


def test_diagnostics_does_not_import_ragas():
    code = "import sys, ragbase.diagnostics; sys.exit(1 if any(m == 'ragas' or m.startswith('ragas.') for m in sys.modules) else 0)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, "导入 ragbase.diagnostics 不应加载 ragas\n" + result.stderr[-500:]


def test_lexical_reexported_from_metrics():
    from ragbase.evaluation import lexical, metrics

    for name in ("normalize_text", "reference_points", "character_ngrams", "lexical_retrieval_metrics",
                 "point_coverage", "POINT_SPLITTER", "NORMALIZE_PATTERN", "LEXICAL_HIT_THRESHOLD"):
        assert getattr(metrics, name) is getattr(lexical, name), name
    assert diagnostics.lexical_retrieval_metrics is lexical.lexical_retrieval_metrics
    assert STAGES == ("embed", "vector_search", "bm25", "fuse", "rerank")


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"通过：{test.__name__}")
    print(f"共 {len(tests)} 项测试通过")
