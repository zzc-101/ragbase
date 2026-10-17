"""运行 Ragas、词面检索指标和原有中文裁判的综合评测。"""
import argparse
import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from config import get_settings
from evaluation_metrics import (
    RagasEvaluator,
    lexical_retrieval_metrics,
    mean_scores,
)
from llm_factory import create_chat_model
from rag_pipeline import (
    ask,
    build_pipeline,
    retrieve_and_rerank,
    select_context_docs,
    should_refuse,
)

load_dotenv()

BASE_DIR = Path(__file__).parent
EVAL_FILE = BASE_DIR / "eval_questions.json"


def result_path(mode: str, retrieval_only: bool) -> Path:
    """按召回模式区分结果文件；仅检索评测另加后缀，避免覆盖完整评测。"""
    suffix = ".retrieval" if retrieval_only else ""
    return BASE_DIR / f"eval_results.{mode}{suffix}.json"


def parse_args():
    settings = get_settings()  # 默认值来自配置（环境变量 / .env），命令行参数可覆盖
    parser = argparse.ArgumentParser(description="评估 RAG 检索与回答质量")
    parser.add_argument("--limit", type=int, help="只评估前 N 道题，适合冒烟测试")
    parser.add_argument("--start-id", type=int, default=1, help="从指定题号开始")
    parser.add_argument(
        "--skip-ragas",
        action="store_true",
        help="跳过 Ragas 指标，减少 API 调用",
    )
    parser.add_argument(
        "--skip-legacy-judge",
        action="store_true",
        help="跳过原有中文 LLM 裁判",
    )
    parser.add_argument(
        "--retrieval-mode",
        choices=("vector", "bm25", "hybrid"),
        default=settings.mode,
        help=f"召回方式：向量 / BM25 / 两路 RRF 融合（默认 {settings.mode}，可由 RAG_MODE 设置）",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="只评估召回与精排的词面指标，不调用 LLM / Ragas / 裁判（不产生 API 费用）",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=settings.min_score,
        help=f"最相关片段的精排得分低于该值则拒答、不调用 LLM（默认 {settings.min_score}；0 表示不拒答）",
    )
    parser.add_argument(
        "--context-min-score",
        type=float,
        default=settings.context_min_score,
        help=f"只把精排得分不低于该值的片段交给 LLM（top1 总是保留；默认 {settings.context_min_score}；0 表示不过滤）",
    )
    return parser.parse_args()


def load_eval_questions(start_id: int = 1, limit: int | None = None):
    questions = json.loads(EVAL_FILE.read_text(encoding="utf-8"))
    questions = [question for question in questions if question["id"] >= start_id]
    return questions[:limit] if limit is not None else questions


def _verdict_lines(text: str) -> list[str]:
    """拆成非空行，并去掉首尾的标点、引号和空白。"""
    strip_chars = " \t\r\n\"'“”‘’「」『』【】[]()（）。.,，、:：;；!！*#`-"
    lines = (line.strip(strip_chars) for line in (text or "").splitlines())
    return [line for line in lines if line]


def _classify_answer(line: str) -> str | None:
    """单行文本 -> 回答裁判标签；先判部分/否定形式，避免“不正确”命中“正确”。"""
    if "部分" in line or "不完全" in line:
        return "部分正确"
    if any(word in line for word in ("不正确", "错误", "不对", "答非所问")):
        return "错误"
    if "正确" in line:
        return "正确"
    return None


def _classify_retrieval(line: str) -> str | None:
    """单行文本 -> 检索裁判标签；先判“不充分/不足”，避免命中“充分”。"""
    if any(word in line for word in ("不充分", "不足", "不够", "几乎没有")):
        return "不足"
    if "部分" in line or "不完全" in line:
        return "部分"
    if "充分" in line:
        return "充分"
    return None


def _first_label(text: str, classify, default: str) -> str:
    """提示词要求只回一个标签：取第一个能识别出标签的非空行，识别不出则取保守标签。"""
    for line in _verdict_lines(text):
        label = classify(line)
        if label:
            return label
    return default


def parse_answer_verdict(text: str) -> str:
    """解析回答裁判输出，返回 正确 / 部分正确 / 错误（无法识别时为 错误）。"""
    return _first_label(text, _classify_answer, "错误")


def parse_retrieval_verdict(text: str) -> str:
    """解析检索裁判输出，返回 充分 / 部分 / 不足（无法识别时为 不足）。"""
    return _first_label(text, _classify_retrieval, "不足")


async def judge_answer(llm, question, answer, reference):
    """保留旧版中文裁判，便于和新的连续分数对照。"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是评测裁判。根据问题和参考要点，判断被评测答案的质量。

参考要点：{reference}
被评测答案：{answer}

- 正确且包含核心内容：回答“正确”
- 方向正确但要点不全或有偏差：回答“部分正确”
- 错误、答非所问或错误地声称没有内容：回答“错误”

只输出“正确”“部分正确”“错误”三者之一，不要输出任何其他文字、标点或解释。"""),
        ("human", "问题：{question}"),
    ])
    result = await (prompt | llm | StrOutputParser()).ainvoke({
        "reference": reference,
        "answer": answer,
        "question": question,
    })
    return parse_answer_verdict(result)


async def check_retrieval(llm, question, candidates, reference):
    """保留旧版中文检索裁判，返回充分、部分或不足。"""
    if not candidates:
        return "不足"  # 没有召回任何片段，无需再调用 LLM
    context = "\n\n".join(document.page_content for document in candidates)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是检索质量评估员。判断召回片段是否足以回答问题。

参考答案要点：{reference}
召回片段：
{context}

- 包含完整所需信息：回答“充分”
- 只有部分相关信息：回答“部分”
- 几乎没有相关信息：回答“不足”

只输出“充分”“部分”“不足”三者之一，不要输出任何其他文字、标点或解释。"""),
        ("human", "问题：{question}"),
    ])
    result = await (prompt | llm | StrOutputParser()).ainvoke({
        "reference": reference,
        "context": context,
        "question": question,
    })
    return parse_retrieval_verdict(result)


def document_record(document):
    source = os.path.basename(document.metadata.get("source", "?"))
    page = document.metadata.get("page")
    return {
        "source": source,
        "page": page + 1 if isinstance(page, int) else None,
        "vec_rank": document.metadata.get("vec_rank"),
        "bm25_rank": document.metadata.get("bm25_rank"),
        "rerank_score": document.metadata.get("rerank_score"),
        "text": document.page_content,
    }


def save_results(results, path: Path):
    """每完成一题就原子保存，长评测中断后仍保留已有结果。"""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def format_scores(scores):
    return " ".join(
        f"{name}={value:.3f}"
        for name, value in scores.items()
        if isinstance(value, (int, float))
    ) or "无可用分数"


async def evaluate_quality(
    ragas_evaluator,
    llm,
    question,
    answer,
    docs,
    candidates,
    reference,
    skip_legacy_judge,
):
    """并发运行 Ragas 和两个旧版裁判，减少评测等待时间。"""
    tasks = {}
    if ragas_evaluator is not None:
        tasks["ragas"] = ragas_evaluator.score(
            question=question,
            response=answer,
            retrieved_contexts=[document.page_content for document in docs],
            reference=reference,
        )
    if not skip_legacy_judge:
        tasks["answer_judge"] = judge_answer(llm, question, answer, reference)
        tasks["retrieval_judge"] = check_retrieval(
            llm,
            question,
            candidates,
            reference,
        )
    if not tasks:
        return None, None

    names = list(tasks)
    values = await asyncio.gather(*tasks.values())
    outcomes = dict(zip(names, values))

    ragas_result = None
    if "ragas" in outcomes:
        scores, errors = outcomes["ragas"]
        ragas_result = {"scores": scores, "errors": errors}

    legacy_judge = None
    if "answer_judge" in outcomes:
        legacy_judge = {
            "answer": outcomes["answer_judge"],
            "retrieval": outcomes["retrieval_judge"],
        }
    return ragas_result, legacy_judge


def print_summary(results, mode: str):
    print("\n" + "=" * 72)
    print(f"评估总结（召回模式：{mode}）")
    print("=" * 72)
    if any(result.get("ragas") for result in results):
        print("Ragas 平均分：", format_scores(mean_scores(results, ("ragas", "scores"))))
    refused = sum(1 for result in results if result.get("refused"))
    print(f"因精排得分过低拒答：{refused}/{len(results)} 题")
    print(
        "召回 Top20 词面代理指标：",
        format_scores(mean_scores(results, ("lexical", "retrieved_top20"))),
    )
    print(
        "精排 Top5 词面代理指标：",
        format_scores(mean_scores(results, ("lexical", "reranked_top5"))),
    )
    sent_counts = [len(result.get("llm_contexts") or []) for result in results]
    print(
        f"交给 LLM 的片段平均 {sum(sent_counts) / len(sent_counts):.2f} 个，词面代理指标：",
        format_scores(mean_scores(results, ("lexical", "llm_context"))),
    )

    verdicts = {}
    retrievals = {}
    for result in results:
        legacy = result.get("legacy_judge") or {}
        verdict = legacy.get("answer")
        retrieval = legacy.get("retrieval")
        if verdict:
            verdicts[verdict] = verdicts.get(verdict, 0) + 1
        if retrieval:
            retrievals[retrieval] = retrievals.get(retrieval, 0) + 1
    if verdicts:
        print("原有回答裁判：", verdicts)
        print("原有检索裁判：", retrievals)


def main():
    args = parse_args()
    settings = get_settings()
    if not args.retrieval_only and not settings.llm_configured:
        raise SystemExit(
            "错误：请在 .env 中配置 RAG_LLM_API_KEY、RAG_LLM_MODEL 和 RAG_LLM_BASE_URL（旧的 ARK_* 变量同样有效）"
        )

    questions = load_eval_questions(args.start_id, args.limit)
    if not questions:
        raise SystemExit("没有符合条件的评测题")

    mode = args.retrieval_mode
    result_file = result_path(mode, args.retrieval_only)
    print(f"加载 {len(questions)} 道测试题，开始评估（召回模式：{mode}）...")
    print("加载本地 embedding 与 reranker 模型...")
    pipeline, _ = build_pipeline(mode=mode)
    # 中文裁判与 ragas 用评测模型（eval_llm_*，缺省沿用回答模型）；回答仍由 pipeline 里的回答模型生成
    judge_llm = None if args.retrieval_only else create_chat_model(settings, "eval")
    ragas_evaluator = None
    if not args.skip_ragas and not args.retrieval_only:
        ragas_evaluator = RagasEvaluator(settings, embeddings=pipeline.embeddings)

    results = []
    with asyncio.Runner() as runner:
        for index, question in enumerate(questions, 1):
            prompt = question["question"]
            reference = question.get("reference", question["answer_keyword"])
            print(f"\n[{index}/{len(questions)}] ID {question['id']}：{prompt}")

            if args.retrieval_only:
                answer, evidence = None, ""
                reranked, candidates = retrieve_and_rerank(pipeline, prompt, settings.rerank_k)
                refused = should_refuse(reranked, args.min_score)
                top_score = reranked[0].metadata["rerank_score"] if reranked else None
                # 与 ask 一致：拒答时没有片段交给 LLM
                sent_docs = [] if refused else select_context_docs(reranked, args.context_min_score)
            else:
                outcome = ask(
                    pipeline,
                    prompt,
                    rerank_k=settings.rerank_k,
                    min_score=args.min_score,
                    context_min_score=args.context_min_score,
                )
                answer, evidence = outcome.answer, outcome.evidence  # answer 已不含【依据】
                reranked, candidates = outcome.reranked, outcome.candidates
                refused, top_score, sent_docs = outcome.refused, outcome.top_score, outcome.docs
            # ragas 上下文用 LLM 实际看到的片段；拒答题没有，退回精排 Top5 以保持原有行为
            docs = sent_docs or reranked
            lexical = {
                "retrieved_top20": lexical_retrieval_metrics(candidates, reference),
                "reranked_top5": lexical_retrieval_metrics(reranked, reference),
                "llm_context": lexical_retrieval_metrics(sent_docs, reference),
            }
            if args.retrieval_only:
                ragas_result, legacy_judge = None, None
            else:
                ragas_result, legacy_judge = runner.run(evaluate_quality(
                    ragas_evaluator=ragas_evaluator,
                    llm=judge_llm,
                    question=prompt,
                    answer=answer,
                    docs=docs,
                    candidates=candidates,
                    reference=reference,
                    skip_legacy_judge=args.skip_legacy_judge,
                ))

            result = {
                "id": question["id"],
                "question": prompt,
                "reference": reference,
                "retrieval_mode": mode,
                "answer": answer,
                "evidence": evidence,
                "top1_rerank_score": top_score,
                "refused": refused,
                "difficulty": question["difficulty"],
                "type": question["type"],
                "lexical": lexical,
                "legacy_judge": legacy_judge,
                "ragas": ragas_result,
                "retrieved_contexts": [document_record(document) for document in reranked],
                "llm_contexts": [document_record(document) for document in sent_docs],
            }
            results.append(result)
            save_results(results, result_file)

            if ragas_result:
                print("  Ragas：", format_scores(ragas_result["scores"]))
                if ragas_result["errors"]:
                    print("  Ragas 错误：", ragas_result["errors"])
            if top_score is not None:
                print(f"  Top1 精排得分：{top_score:.3f}{'（会被拒答）' if refused else ''}")
            print("  词面 Top20：", format_scores(lexical["retrieved_top20"]))
            print("  词面 Top5： ", format_scores(lexical["reranked_top5"]))
            print(
                f"  词面 LLM 实际片段（{len(sent_docs)} 个）：",
                format_scores(lexical["llm_context"]),
            )
            if legacy_judge:
                print(
                    f"  原有裁判：回答={legacy_judge['answer']} "
                    f"检索={legacy_judge['retrieval']}"
                )

    print_summary(results, mode)
    print(f"\n详细结果已保存到 {result_file}")


if __name__ == "__main__":
    main()
