"""ragbase diagnose：RAG 参数与速度体检（只测量和给建议，不改配置、不扫参数、不调用大模型）。

流程：从数据库读评测题集 -> 构建流水线（记录模型加载耗时）-> 用固定假问题预热 -> 对每道题分阶段计时地跑召回 + 精排，
算三组词面检索指标 -> 汇总 -> 按固定规则生成建议 -> 终端打印报告，并把完整数据存进数据库的报告历史（可选 --json 导出）。

代码分三层，便于不加载模型就能单测：
- 纯函数：summarize（统计汇总）、make_recommendations（规则生成建议）、render_report（排版）、chunk_length_stats 等；
- measure_question / collect_params / run_diagnosis：依赖 pipeline / settings 的鸭子类型（假对象即可测试）；
  run_diagnosis 是体检核心（命令行和 HTTP API 共用），不打印任何东西；
- run：命令行入口（加载真实模型、读写数据库、打印），只在 `ragbase diagnose` 里用。

只导入轻量模块（evaluation.lexical 而非 evaluation.metrics），不会加载 ragas。
"""
import hashlib
import json
import statistics
import time
import unicodedata
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from ragbase.config import get_settings
from ragbase.evaluation.lexical import lexical_retrieval_metrics, reference_points
from ragbase.ingest.splitter import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from ragbase.retrieval.pipeline import (
    build_pipeline,
    retrieve_and_rerank,
    select_context_docs,
    should_refuse,
)

# ---------------------------------------------------------------- 建议规则的阈值（初始值，可按经验调整）

RERANK_SHARE_PCT = 60.0  # 规则 a：精排占检索总耗时（召回 + 精排）的百分比达到该值，认为精排是速度瓶颈
CANDIDATE_RECALL_LOW = 0.6  # 规则 c：召回 TopK 候选的 point_recall 均值低于该值，认为召回阶段漏掉了答案
RERANK_RECALL_DROP = 0.15  # 规则 d：召回 TopK 比精排后的 point_recall 高出该值以上，认为精排把相关片段排出了保留范围
CONTEXT_RECALL_DROP = 0.10  # 规则 e：精排后比交给大模型的片段的 point_recall 高出该值以上，认为上下文过滤偏严
REFUSAL_RATE_HIGH = 0.20  # 规则 f：拒答率（占参与统计的题数）高于该值，认为拒答阈值可能偏高
CLOUD_LATENCY_MS = 1000.0  # 规则 h：云端 embedding / 精排单次耗时均值超过该毫秒数，认为网络延迟偏大
MIN_QUESTIONS_FOR_QUALITY_RULES = 5  # 质量类规则（c / d / e / f）需要的最少题数：样本太少时均值不稳，拿不准就不给建议

WARMUP_QUESTION = "预热用的占位问题：请忽略这条"  # 不在题集里的固定假问题，正式计时前跑一遍，不计入统计
STAGES = ("embed", "vector_search", "bm25", "fuse", "rerank")  # 与 HybridRetriever.invoke / retrieve_and_rerank 写入 timings 的键一致
STAGE_LABELS = {
    "embed": "向量化查询（embed）",
    "vector_search": "向量库查询",
    "bm25": "BM25 关键词（含分词）",
    "fuse": "RRF 融合 + 组装",
    "rerank": "精排（rerank）",
}
METRIC_NAMES = ("point_recall", "context_precision", "mrr")
GROUP_LABELS = {
    "candidates": "召回 TopK 候选",
    "reranked": "精排后",
    "llm_context": "交给大模型的片段",
}


# ---------------------------------------------------------------- 题集

def reference_of(question: dict) -> str | None:
    """题目的参考答案（reference 字段）；没有，或拆不出任何要点时返回 None（这种题只计耗时，不计召回率）。"""
    reference = question.get("reference")
    if not isinstance(reference, str) or not reference_points(reference):
        return None
    return reference


def dedupe_questions(questions: list[dict]) -> tuple[list[dict], int]:
    """同文本（去掉首尾空白）的问题只保留第一道，返回 (去重后的题, 被跳过的数量)。

    embedding 和精排都有进程内缓存：同一个问题第二次几乎是 0 毫秒，会让耗时统计失真。
    """
    seen, unique = set(), []
    for question in questions:
        key = question["question"].strip()
        if key in seen:
            continue
        seen.add(key)
        unique.append(question)
    return unique, len(questions) - len(unique)


def load_questions(store, limit: int | None = None) -> tuple[list[dict], int]:
    """从题集（QuestionStore）读题，取前 limit 道后去重，返回 (题, 因重复被跳过的数量)。

    题集为空时给出中文说明后退出（SystemExit），不抛 traceback。
    """
    questions = store.list()
    if limit is not None:
        questions = questions[:limit]
    if not questions:
        raise SystemExit("错误：评测题集是空的。请在 Web 的体检页添加题目，或运行：ragbase questions import <文件>（格式见 docs/evaluation.md#评测题格式）")
    return dedupe_questions(questions)


def digest_questions(questions: list[dict]) -> str:
    """题集内容摘要：按 id 排序后，对每题的 `id|question|reference` 做 md5。与题目顺序无关、与内容有关，用来判断两份报告的题集是否相同。"""
    ordered = sorted(questions, key=lambda q: (q.get("id") is None, q.get("id") or 0))
    text = "\n".join(f"{q.get('id')}|{q['question']}|{q.get('reference') or ''}" for q in ordered)
    return hashlib.md5(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- 测量

def measure_question(
    pipeline,
    question: dict,
    rerank_k: int,
    min_score: float | None,
    context_min_score: float | None,
    mode: str | None = None,
) -> dict:
    """对一道题跑一遍召回 + 精排（不调用大模型；mode 为 None 时用检索器的默认召回方式），返回逐题记录：各阶段耗时（毫秒）、候选数、拒答、三组词面指标。

    三组指标的口径与 `ragbase eval --retrieval-only` 完全一致：召回候选、精排后、交给大模型的片段
    （拒答时没有片段交给大模型）。没有参考答案的题 lexical 为 None，只计耗时。
    """
    text = question["question"]
    timings: dict[str, float] = {}
    start = time.perf_counter()
    reranked, candidates = retrieve_and_rerank(pipeline, text, rerank_k, mode=mode, timings=timings)
    total = time.perf_counter() - start

    refused = should_refuse(reranked, min_score)
    sent_docs = [] if refused else select_context_docs(reranked, context_min_score)
    top_score = reranked[0].metadata.get("rerank_score") if reranked else None
    reference = reference_of(question)
    lexical = None
    if reference is not None:
        lexical = {
            "candidates": lexical_retrieval_metrics(candidates, reference),
            "reranked": lexical_retrieval_metrics(reranked, reference),
            "llm_context": lexical_retrieval_metrics(sent_docs, reference),
        }
    timings_ms = {name: round(timings[name] * 1000, 3) for name in STAGES if name in timings}
    timings_ms["total"] = round(total * 1000, 3)
    return {
        "id": question.get("id"),
        "question": text,
        "has_reference": reference is not None,
        "timings_ms": timings_ms,
        "candidate_count": len(candidates),
        "reranked_count": len(reranked),
        "sent_count": len(sent_docs),
        "top1_rerank_score": top_score,
        "refused": refused,
        "lexical": lexical,
    }


# ---------------------------------------------------------------- 统计汇总（纯函数）

def _stats(values: list[float], digits: int = 3) -> dict | None:
    """均值 / 中位数 / 最大值 / 最小值 / 个数；没有数据返回 None。"""
    if not values:
        return None
    return {
        "mean": round(statistics.fmean(values), digits),
        "median": round(statistics.median(values), digits),
        "max": round(max(values), digits),
        "min": round(min(values), digits),
        "count": len(values),
    }


def _mean(values: list[float], digits: int = 4) -> float | None:
    return round(statistics.fmean(values), digits) if values else None


def chunk_length_stats(texts: list[str]) -> dict | None:
    """片段长度（字符数）统计：片段数 / 均值 / 中位数 / 最大值；没有片段返回 None。"""
    if not texts:
        return None
    lengths = [len(text) for text in texts]
    return {
        "count": len(lengths),
        "mean": round(statistics.fmean(lengths), 1),
        "median": round(statistics.median(lengths), 1),
        "max": max(lengths),
    }


def summarize(records: list[dict], duplicates_skipped: int = 0) -> dict:
    """把逐题记录汇总成：题数、质量（三组指标均值、拒答率、Top1 得分分布、平均片段数）、速度（各阶段统计与占比）。"""
    measured = len(records)
    reference_records = [record for record in records if record.get("lexical")]

    quality: dict = {}
    for group in GROUP_LABELS:
        quality[group] = (
            {name: _mean([record["lexical"][group][name] for record in reference_records]) for name in METRIC_NAMES}
            if reference_records else None
        )
    # 上下文过滤造成的召回损失：只看没被拒答的有参考答案的题（拒答题没有片段交给大模型，其损失属于拒答而不是过滤）
    filtered = [record for record in reference_records if not record["refused"]]
    quality["context_filter"] = {
        "questions": len(filtered),
        "reranked_point_recall": _mean([record["lexical"]["reranked"]["point_recall"] for record in filtered]),
        "llm_point_recall": _mean([record["lexical"]["llm_context"]["point_recall"] for record in filtered]),
        "gap": _mean([
            record["lexical"]["reranked"]["point_recall"] - record["lexical"]["llm_context"]["point_recall"]
            for record in filtered
        ]),
    } if filtered else None
    refused = sum(1 for record in records if record["refused"])
    quality["refused"] = refused
    quality["refusal_rate"] = round(refused / measured, 4) if measured else 0.0
    quality["top1_score"] = _stats(
        [record["top1_rerank_score"] for record in records if record.get("top1_rerank_score") is not None], 4
    )
    quality["avg_sent_docs"] = _mean([record["sent_count"] for record in records], 2)

    total_ms = [record["timings_ms"]["total"] for record in records]
    total_sum = sum(total_ms)
    stages: dict = {}
    for name in STAGES:
        values = [record["timings_ms"][name] for record in records if name in record["timings_ms"]]
        stat = _stats(values)
        if stat is not None:
            stat["share_pct"] = round(sum(values) / total_sum * 100, 1) if total_sum > 0 else 0.0
            stages[name] = stat
    rerank_records = [record for record in records if "rerank" in record["timings_ms"] and record["candidate_count"] > 0]
    candidate_sum = sum(record["candidate_count"] for record in rerank_records)
    speed = {
        "total": _stats(total_ms),
        "stages": stages,
        "avg_candidates": _mean([record["candidate_count"] for record in records], 2),
        # 精排每条候选平均毫秒 = 精排总时间 / 候选总条数
        "rerank_ms_per_candidate": (
            round(sum(record["timings_ms"]["rerank"] for record in rerank_records) / candidate_sum, 3)
            if candidate_sum else None
        ),
    }
    return {
        "questions": {
            "measured": measured,
            "with_reference": len(reference_records),
            "duplicates_skipped": duplicates_skipped,
        },
        "quality": quality,
        "speed": speed,
    }


# ---------------------------------------------------------------- 建议规则（纯函数）

def _recommendation(rule_id: str, title: str, phenomenon: str, attempt: str, cost: str) -> dict:
    return {"id": rule_id, "title": title, "phenomenon": phenomenon, "try": attempt, "cost": cost}


def make_recommendations(data: dict) -> list[dict]:
    """按固定规则从汇总数据生成建议（不是定论）。

    data = {"params": collect_params 的结果, "summary": summarize 的结果}；只读这两个 dict，不依赖模型。
    返回 [{id, title, phenomenon, try, cost}]，没有规则触发时返回 []。
    """
    params, summary = data["params"], data["summary"]
    quality, speed, counts = summary["quality"], summary["speed"], summary["questions"]
    stages = speed["stages"]
    mode = params["mode"]
    embedding_local = params["embedding_provider"] == "local"
    reranker_provider = params["reranker_provider"]
    has_reference = counts["with_reference"] >= MIN_QUESTIONS_FOR_QUALITY_RULES
    items: list[dict] = []

    # a. 精排拖慢检索
    rerank = stages.get("rerank")
    if rerank and rerank["share_pct"] >= RERANK_SHARE_PCT:
        per_candidate = speed.get("rerank_ms_per_candidate")
        detail = f"，平均每条候选 {per_candidate:.1f} ms" if per_candidate is not None else ""
        attempt = f"减少精排候选数 RAG_RETRIEVE_K（当前 {params['retrieve_k']}）"
        if reranker_provider == "local":
            attempt += (
                "；换更小的精排模型 RAG_RERANKER_MODEL（需重新校准 RAG_MIN_SCORE）"
                "；改用云端精排 RAG_RERANKER_PROVIDER=api"
            )
            if params["device"] == "cpu":
                attempt += "；有 GPU / Apple 芯片时用 RAG_DEVICE 指定"
            cost = "召回 TopK 的命中率可能下降；云端精排会把文本发往外部服务并产生费用"
        else:
            attempt += "；换响应更快的精排模型或就近的服务（换精排模型需重新校准 RAG_MIN_SCORE）"
            cost = "召回 TopK 的命中率可能下降；换模型后精排得分分布会变，要重新校准拒答阈值"
        items.append(_recommendation(
            "rerank_slow", "精排是速度瓶颈",
            f"精排平均 {rerank['mean']:.0f} ms，占检索总耗时的 {rerank['share_pct']:.0f}%（阈值 {RERANK_SHARE_PCT:.0f}%）{detail}",
            attempt, cost,
        ))

    # b. 本地模型跑在 CPU 上
    if (embedding_local or reranker_provider == "local") and params["device"] == "cpu":
        items.append(_recommendation(
            "cpu_device", "本地模型在 CPU 上运行",
            "本地模型的计算设备是 cpu，通常比 GPU / Apple 芯片（mps）慢一个数量级",
            "有 GPU / Apple 芯片时用 RAG_DEVICE=cuda 或 mps（默认 auto 会自动选择，若仍是 cpu 说明没检测到可用设备）；"
            "或把 embedding / 精排改用云端（RAG_EMBEDDING_PROVIDER=openai、RAG_RERANKER_PROVIDER=api）",
            "云端方案会把文本发往外部服务并产生费用，且延迟含网络往返；换 embedding 必须 ragbase ingest --full 重建索引",
        ))

    # c. 召回阶段漏掉答案
    candidates = quality.get("candidates")
    if candidates and has_reference and candidates["point_recall"] < CANDIDATE_RECALL_LOW:
        attempt_parts = []
        if mode != "hybrid":
            attempt_parts.append(f"改用 hybrid 召回（当前 {mode}；--retrieval-mode hybrid 或 RAG_MODE=hybrid）")
        attempt_parts.append(
            f"增大 RAG_PER_ROUTE_K（当前 {params['per_route_k']}）/ RAG_RETRIEVE_K（当前 {params['retrieve_k']}）"
        )
        attempt_parts.append("检查文档切分是否合适、BM25 用户词典 bm25_userdict.txt 是否收录了领域术语")
        items.append(_recommendation(
            "low_candidate_recall", "召回阶段漏掉了答案",
            f"召回 TopK 候选的 point_recall 均值 {candidates['point_recall']:.2f}（阈值 {CANDIDATE_RECALL_LOW:.2f}），"
            "说明答案要点在进入精排前就没被召回",
            "；".join(attempt_parts),
            "候选更多，精排更慢（精排耗时基本与候选数成正比）",
        ))

    # d. 精排把相关片段排出了保留范围
    reranked = quality.get("reranked")
    if candidates and reranked and has_reference:
        drop = round(candidates["point_recall"] - reranked["point_recall"], 4)  # 先取整，避免浮点误差影响边界判断
        if drop > RERANK_RECALL_DROP:
            attempt = f"增大 RAG_RERANK_K（当前 {params['rerank_k']}）"
            if reranker_provider != "none":
                attempt += "；或换更准的精排模型 RAG_RERANKER_MODEL（需重新校准 RAG_MIN_SCORE）"
            items.append(_recommendation(
                "rerank_drops_relevant", "精排把相关片段排出了保留范围",
                f"召回 TopK 的 point_recall {candidates['point_recall']:.2f}，精排后降到 {reranked['point_recall']:.2f}"
                f"（相差 {drop:.2f}，阈值 {RERANK_RECALL_DROP:.2f}）",
                attempt,
                "交给大模型的上下文变长，大模型费用和耗时增加",
            ))

    # e. 上下文过滤偏严（只比较没被拒答的题，避免把拒答造成的损失算到过滤头上）
    context_filter = quality.get("context_filter")
    if context_filter and context_filter["questions"] >= MIN_QUESTIONS_FOR_QUALITY_RULES and context_filter["gap"] > CONTEXT_RECALL_DROP:
        items.append(_recommendation(
            "context_filter_strict", "上下文过滤偏严",
            f"（排除拒答题）精排后的 point_recall {context_filter['reranked_point_recall']:.2f}，"
            f"过滤后交给大模型的片段降到 {context_filter['llm_point_recall']:.2f}"
            f"（相差 {context_filter['gap']:.2f}，阈值 {CONTEXT_RECALL_DROP:.2f}）",
            f"降低 RAG_CONTEXT_MIN_SCORE（当前 {params['context_min_score']}）",
            "交给大模型的片段更多，可能混入相关度较低的内容，大模型费用增加",
        ))

    # f. 拒答率偏高（没有精排时只有空召回才拒答，不适用阈值调整）
    if (
        reranker_provider != "none"
        and counts["measured"] >= MIN_QUESTIONS_FOR_QUALITY_RULES
        and quality["refusal_rate"] > REFUSAL_RATE_HIGH
    ):
        items.append(_recommendation(
            "high_refusal", "拒答率偏高",
            f"{quality['refused']}/{counts['measured']} 题（{quality['refusal_rate'] * 100:.0f}%）会被拒答，阈值 {REFUSAL_RATE_HIGH * 100:.0f}%",
            f"如果这些题的答案都应该在库里，说明 RAG_MIN_SCORE（当前 {params['min_score']}）可能偏高；"
            "运行 scripts/calibrate_rerank_threshold.py 重新校准",
            "阈值调低后，库外问题更容易漏拦（召回无关片段后仍交给大模型作答）",
        ))

    # g. 没有精排
    if reranker_provider == "none":
        items.append(_recommendation(
            "no_reranker", "未启用精排",
            "RAG_RERANKER_PROVIDER=none：片段保持召回顺序，没有精排得分",
            "没有精排就不会按相关度拒答（只有空召回才拒答）；需要拒答能力请改用 local 或 api 精排并校准 RAG_MIN_SCORE",
            "精排会增加耗时（本地模型）或产生 API 费用（云端）",
        ))

    # h. 云端调用延迟偏大
    slow = []
    embed = stages.get("embed")
    if not embedding_local and embed and embed["mean"] > CLOUD_LATENCY_MS:
        slow.append(f"云端 embedding 单次均值 {embed['mean']:.0f} ms")
    if reranker_provider == "api" and rerank and rerank["mean"] > CLOUD_LATENCY_MS:
        slow.append(f"云端精排单次均值 {rerank['mean']:.0f} ms")
    if slow:
        items.append(_recommendation(
            "cloud_latency", "云端调用延迟偏大",
            f"{'；'.join(slow)}（阈值 {CLOUD_LATENCY_MS:.0f} ms），耗时主要是网络往返",
            "换用就近的云端服务（RAG_EMBEDDING_BASE_URL / RAG_RERANKER_BASE_URL），或改用本地模型",
            "本地模型要占用显存 / 内存，CPU 上更慢；换 embedding 必须 ragbase ingest --full 重建索引",
        ))
    return items


# ---------------------------------------------------------------- 当前参数

def collect_chunk_texts(retriever) -> list[str]:
    """尽量便宜地拿到全部片段正文：BM25 快照里有语料就直接用，否则遍历向量库。拿不到返回 []。"""
    snapshot = getattr(retriever, "_snapshot", None)
    corpus = getattr(snapshot, "corpus", None)
    if corpus:
        return [text for text, _ in corpus.values()]
    store = getattr(retriever, "store", None)
    if store is None:
        return []
    return [record.text for record in store.iter_all()]


def collect_params(settings, pipeline, mode: str, load_seconds: float, chunk_texts: list[str]) -> dict:
    """当前生效的参数与环境，写进报告第一部分（settings / pipeline 用鸭子类型，测试里可传假对象）。"""
    reranker_provider = settings.reranker_provider
    batch_size = getattr(pipeline.reranker, "_batch_size", None) if reranker_provider == "local" else None
    return {
        "mode": mode,
        "per_route_k": settings.per_route_k,
        "retrieve_k": settings.retrieve_k,
        "rerank_k": settings.rerank_k,
        "min_score": settings.min_score,
        "context_min_score": settings.context_min_score,
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.embedding_model,
        "reranker_provider": reranker_provider,
        "reranker_model": settings.reranker_model if reranker_provider != "none" else None,
        "reranker_batch_size": batch_size,
        "device": pipeline.device or "未知",
        "chunks": chunk_length_stats(chunk_texts),
        "default_chunk_size": DEFAULT_CHUNK_SIZE,
        "default_chunk_overlap": DEFAULT_CHUNK_OVERLAP,
        "model_load_seconds": round(load_seconds, 2),
    }


# ---------------------------------------------------------------- 终端排版

def _width(text: str) -> int:
    """终端显示宽度：中文等全角字符占 2 列。"""
    return sum(2 if unicodedata.east_asian_width(char) in "WF" else 1 for char in text)


def _pad(text: str, width: int) -> str:
    return text + " " * max(width - _width(text), 0)


def _rpad(text: str, width: int) -> str:
    """右对齐版 _pad：表头含中文时 f-string 的 :>N 按字符数算，会错位。"""
    return " " * max(width - _width(text), 0) + text


def _fmt_ms(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f}"


def render_report(report: dict) -> str:
    """把报告 dict（params / summary / recommendations）排成可直接打印的中文文本。"""
    params, summary, recommendations = report["params"], report["summary"], report["recommendations"]
    counts, quality, speed = summary["questions"], summary["quality"], summary["speed"]
    mode = params["mode"]
    lines: list[str] = []
    bar = "=" * 72

    def kv(label: str, value) -> None:
        lines.append(f"  {_pad(label, 36)}{value}")

    lines += [bar, f"RAG 体检报告（召回方式：{mode}）", bar, "", "【当前参数】"]
    kv("召回方式", mode)
    kv("per_route_k（混合每路召回数）", f"{params['per_route_k']}" + ("" if mode == "hybrid" else "（仅 hybrid 模式生效）"))
    kv("retrieve_k（精排候选数）", params["retrieve_k"])
    kv("rerank_k（精排后保留数）", params["rerank_k"])
    kv("min_score（低于则拒答）", params["min_score"])
    kv("context_min_score（上下文过滤）", params["context_min_score"])
    kv("embedding", f"{params['embedding_provider']} / {params['embedding_model']}")
    if params["reranker_provider"] == "none":
        kv("reranker", "none（不精排）")
    else:
        kv("reranker", f"{params['reranker_provider']} / {params['reranker_model']}")
    batch = params["reranker_batch_size"]
    kv("reranker 批大小", batch if batch is not None else "不适用（非本地精排）")
    kv("计算设备", params["device"])
    chunks = params["chunks"]
    if chunks:
        kv("片段数", chunks["count"])
        kv("片段长度（字符）", f"均值 {chunks['mean']:.0f} / 中位数 {chunks['median']:.0f} / 最大 {chunks['max']}")
    else:
        kv("片段数", "未能读取")
    kv("切分默认值", f"chunk_size={params['default_chunk_size']}，overlap={params['default_chunk_overlap']}（代码默认，非索引实测）")
    kv("模型加载耗时", f"{params['model_load_seconds']:.1f} s（含 BM25 索引快照构建）")
    duplicates = f"，另有 {counts['duplicates_skipped']} 道重复题已跳过" if counts["duplicates_skipped"] else ""
    kv("参与统计的题数", f"{counts['measured']}（其中有参考答案 {counts['with_reference']}）{duplicates}")

    lines += ["", "【质量】（词面代理指标，仅在同一题集上做相对比较）"]
    if quality["candidates"] is None:
        lines.append("  没有带参考答案的题，无法计算召回指标。")
    else:
        header = f"  {_pad('阶段', 22)}{'point_recall':>14}{'context_precision':>20}{'mrr':>8}"
        lines.append(header)
        for group, label in GROUP_LABELS.items():
            row = quality[group]
            lines.append(
                f"  {_pad(label, 22)}{row['point_recall']:>14.3f}{row['context_precision']:>20.3f}{row['mrr']:>8.3f}"
            )
    kv("拒答率", f"{quality['refusal_rate'] * 100:.1f}%（{quality['refused']}/{counts['measured']}）")
    top1 = quality["top1_score"]
    if top1:
        kv("Top1 精排得分", f"最小 {top1['min']:.3f} / 中位数 {top1['median']:.3f} / 最大 {top1['max']:.3f}")
    else:
        kv("Top1 精排得分", "无（未启用精排，或没有召回结果）")
    kv("平均交给大模型的片段数", f"{quality['avg_sent_docs']:.2f}" if quality["avg_sent_docs"] is not None else "-")

    lines += ["", "【速度】（毫秒，每题一次召回 + 精排）"]
    concurrent = report.get("concurrent_chat_requests") or 0
    if concurrent > 0:
        lines.append(f"  注意：体检期间有 {concurrent} 次对话请求，耗时可能偏高。")
    lines.append(f"  {_pad('阶段', 24)}{_rpad('均值', 9)}{_rpad('中位数', 9)}{_rpad('最大', 9)}{_rpad('占比', 8)}  备注")
    network = {
        "embed": "含网络往返" if params["embedding_provider"] != "local" else "",
        "rerank": "含网络往返" if params["reranker_provider"] == "api" else "",
    }
    for name in STAGES:
        stat = speed["stages"].get(name)
        if stat is None:
            continue
        lines.append(
            f"  {_pad(STAGE_LABELS[name], 24)}{_fmt_ms(stat['mean']):>9}{_fmt_ms(stat['median']):>9}"
            f"{_fmt_ms(stat['max']):>9}{stat['share_pct']:>7.1f}%  {network.get(name, '')}"
        )
    total = speed["total"]
    if total:
        lines.append(
            f"  {_pad('检索总耗时（召回+精排）', 24)}{_fmt_ms(total['mean']):>9}{_fmt_ms(total['median']):>9}"
            f"{_fmt_ms(total['max']):>9}{'100.0%':>8}"
        )
    per_candidate = speed["rerank_ms_per_candidate"]
    if per_candidate is not None:
        suffix = "（含网络往返，云端精排按批计费，该值仅供参考）" if params["reranker_provider"] == "api" else ""
        kv("精排每条候选平均", f"{per_candidate:.2f} ms（平均 {speed['avg_candidates']:.1f} 条候选）{suffix}")
    lines.append("  各阶段占比相加可能略低于 100%：差额是阶段之间的少量调度开销。")

    lines += ["", "【建议】（基于以上数据按固定规则生成，是参考而不是定论）"]
    if not recommendations:
        lines.append("  未发现明显问题。")
    for number, item in enumerate(recommendations, 1):
        lines.append(f"  {number}. {item['title']}")
        lines.append(f"     现象：{item['phenomenon']}")
        lines.append(f"     可以尝试：{item['try']}")
        lines.append(f"     可能的代价：{item['cost']}")
    lines += [
        "",
        "提示：以上都是基于当前题集的相对指标，不是绝对质量；调整参数后重新运行 ragbase diagnose 对比。",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------- 报告文件

def build_report(
    params: dict,
    records: list[dict],
    duplicates_skipped: int,
    generated_at: str | None = None,
    question_ids: list | None = None,
    question_digest: str | None = None,
    concurrent_chat_requests: int = 0,
) -> dict:
    """组装完整报告 dict：参数、逐题记录、汇总、触发的建议，外加所用题目的 id 列表与内容摘要、体检期间的对话请求数。

    question_ids 缺省取逐题记录里的 id；question_digest 缺省为 None（run_diagnosis 会用 digest_questions 算好传入）。
    """
    summary = summarize(records, duplicates_skipped)
    return {
        "generated_at": generated_at or datetime.now().isoformat(timespec="seconds"),
        "params": params,
        "warmup_question": WARMUP_QUESTION,
        "questions": records,
        "summary": summary,
        "recommendations": make_recommendations({"params": params, "summary": summary}),
        "question_ids": list(question_ids) if question_ids is not None else [record.get("id") for record in records],
        "question_digest": question_digest,
        "concurrent_chat_requests": concurrent_chat_requests,
    }


def save_report(report: dict, path: Path) -> None:
    """原子写入 JSON（先写 .tmp 再替换），中断时不会留下写了一半的报告。"""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


# ---------------------------------------------------------------- 体检核心

def run_diagnosis(
    pipeline,
    settings,
    questions: list[dict],
    mode: str,
    load_seconds: float,
    duplicates_skipped: int,
    on_progress: Callable[[int, int], None] | None = None,
    chat_requests: Callable[[], int] | None = None,
) -> dict:
    """先用固定假问题预热（不计入统计），再逐题测量，返回 build_report 的结果。不打印任何东西。

    questions 是已去重的题（用开始时的快照，运行中题集被编辑不受影响）；on_progress(已完成题数, 总题数) 每完成一题调用一次；
    chat_requests 返回服务累计处理的对话请求数，报告里的 concurrent_chat_requests = 结束时的值 - 开始时的值（None 记 0）。
    测量过程中的异常（云端接口失败等）原样抛出，由调用方处理。
    """
    chats_before = chat_requests() if chat_requests is not None else 0
    retrieve_and_rerank(pipeline, WARMUP_QUESTION, settings.rerank_k, mode=mode)
    records: list[dict] = []
    for index, question in enumerate(questions, 1):
        records.append(measure_question(
            pipeline, question, settings.rerank_k, settings.min_score, settings.context_min_score, mode=mode
        ))
        if on_progress is not None:
            on_progress(index, len(questions))
    concurrent = (chat_requests() - chats_before) if chat_requests is not None else 0
    params = collect_params(settings, pipeline, mode, load_seconds, collect_chunk_texts(pipeline.retriever))
    return build_report(
        params, records, duplicates_skipped,
        question_ids=[question.get("id") for question in questions],
        question_digest=digest_questions(questions),
        concurrent_chat_requests=concurrent,
    )


# ---------------------------------------------------------------- 命令行入口

def _build_pipeline_or_exit(settings, mode):
    """构建流水线；embedding 配置与索引不一致 / 云端提供方配置有误时给出说明后退出，而不是抛 traceback。

    与 evaluation.runner._build_pipeline_or_exit 相同；不直接复用它是因为 runner 顶层会导入 ragas（很重）。
    """
    from ragbase.index_info import IndexMismatchError
    from ragbase.models.factory import ProviderConfigError

    try:
        return build_pipeline(mode=mode, settings=settings)[0]
    except (IndexMismatchError, ProviderConfigError) as exc:
        raise SystemExit(f"错误：{exc}")


def run(args) -> None:
    """运行体检；args 是 `ragbase diagnose` 解析出的命令行参数（见 ragbase.cli）。"""
    from ragbase.db import DatabaseError, open_database
    from ragbase.diagnostics_reports import ReportStore
    from ragbase.evaluation.questions import QuestionStore

    settings = get_settings()
    mode = args.retrieval_mode or settings.mode
    if args.limit is not None and args.limit < 1:
        raise SystemExit("错误：--limit 必须是正整数")
    try:
        db = open_database(settings)
    except DatabaseError as exc:
        raise SystemExit(f"错误：{exc}")
    try:
        questions, duplicates = load_questions(QuestionStore(db), args.limit)

        from ragbase.ingest.common import collection_count

        if not collection_count(settings):
            raise SystemExit("错误：索引不存在或为空，请先运行：ragbase ingest --full")

        note = f"（另有 {duplicates} 道重复题已跳过）" if duplicates else ""
        print(f"加载 {len(questions)} 道题{note}，开始体检（召回方式：{mode}）；不会调用大模型。")
        print("加载 embedding 与 reranker（本地模型或云端接口，按配置）...")
        start = time.perf_counter()
        pipeline = _build_pipeline_or_exit(settings, mode)
        load_seconds = time.perf_counter() - start
        print(f"加载完成，用时 {load_seconds:.1f}s。")

        print("预热一次（不计入统计），然后逐题测量...")
        try:
            report = run_diagnosis(
                pipeline, settings, questions, mode, load_seconds, duplicates,
                on_progress=lambda done, total: print(f"  [{done}/{total}]"),
            )
        except KeyboardInterrupt:
            raise SystemExit("\n已中断，未生成报告。")
        except Exception as exc:  # 云端接口失败、BM25 未构建等：给出提示而不是 traceback
            raise SystemExit(f"错误：测量过程中检索失败：{type(exc).__name__}: {exc}")

        report_id = ReportStore(db).save(report, "cli")
    finally:
        db.close()
    print("\n" + render_report(report))
    print(f"\n报告 id：{report_id}（可在 Web 体检页查看）")
    if args.json:
        path = Path(args.json).expanduser()
        try:
            save_report(report, path)
        except OSError as exc:
            raise SystemExit(f"错误：报告已存入数据库（id {report_id}），但无法写入 {path}：{exc}")
        print(f"完整数据已导出到 {path}")
