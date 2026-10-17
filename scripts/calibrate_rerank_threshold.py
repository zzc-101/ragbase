"""校准拒答阈值 RERANK_MIN_SCORE（只跑召回 + 精排，不调用 LLM / API）。

对比两组问题的最相关片段得分（top1 rerank_score，即 P(相关)）：
  (a) 库内可回答：eval_questions.json 的 30 道题 + 下面 EXTRA_ANSWERABLE 的口语化问题，
      阈值应让它们全部通过；
  (b) 本脚本里的库外问题：库里没有答案，阈值应尽量把它们拦下。

运行（在项目主目录，即 RAG_HOME，默认当前目录下）：uv run python scripts/calibrate_rerank_threshold.py [--mode hybrid]
"""
import argparse
import json
import statistics

from ragbase.config import eval_questions_file, load_env
from ragbase.retrieval.pipeline import build_pipeline, retrieve_and_rerank

load_env()

CANDIDATE_THRESHOLDS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
SAFETY_MARGIN = 0.75  # 推荐阈值 = 可回答题最低分 * 该系数，给没测到的题留余量

# 口语化、简短的库内问题（教材里有对应内容），补充评测集偏“标准考题”的不足
EXTRA_ANSWERABLE = [
    "挣值分析怎么算", "什么是关键路径", "TOGAF 是什么", "CMMI 有几个等级",
    "UML 有哪些图", "耦合和内聚有什么区别", "需求变更怎么控制", "RUP 分哪几个阶段",
    "什么是SPI", "项目干系人有哪些", "怎么写好一个WBS", "软件体系结构是什么意思",
    "讲讲MVC", "ISO 9001是什么", "信息系统安全有哪些要求",
]

# (类型, 问题)：库外问题，教材（信息系统项目管理师 / 系统架构设计师）里不会有答案
OUT_OF_SCOPE = [
    # 完全无关
    ("无关", "红烧肉怎么做才好吃？"),
    ("无关", "明天北京天气怎么样？"),
    ("无关", "世界杯历史上谁进球最多？"),
    ("无关", "周杰伦有哪些经典专辑？"),
    ("无关", "怎么才能快速减肥？"),
    ("无关", "月球距离地球有多远？"),
    ("无关", "推荐几部好看的科幻电影"),
    # 领域相邻，但教材大概率没有
    ("相邻", "Python 3.13 新增了哪些特性？"),
    ("相邻", "Kubernetes 的 Pod 是如何调度的？"),
    ("相邻", "ChatGPT 是哪家公司开发的？"),
    ("相邻", "2024 年软考考试时间是哪天？"),
    ("相邻", "我的项目经理叫什么名字？"),
    ("相邻", "我们公司的项目预算是多少？"),
    ("相邻", "React 和 Vue 哪个更适合做后台管理系统？"),
    # 与语料共享词汇，但问的内容教材答不了
    ("似是而非", "项目经理每月工资一般是多少？"),
    ("似是而非", "软件架构师需要考哪些国外认证，费用多少？"),
    ("似是而非", "微服务架构在亚马逊内部是哪年开始推行的？"),
    ("似是而非", "项目管理软件 Jira 的价格是多少？"),
    ("似是而非", "敏捷开发中 Scrum Master 的年薪是多少？"),
    ("似是而非", "数据仓库产品 Snowflake 与 Redshift 哪个更便宜？"),
    ("似是而非", "瀑布模型是谁在哪一年的哪次会议上提出的？"),
]


def load_answerable() -> list[str]:
    questions = json.loads(eval_questions_file().read_text(encoding="utf-8"))
    return [item["question"] for item in questions] + EXTRA_ANSWERABLE


def top1_score(pipeline, question: str) -> float:
    docs, _ = retrieve_and_rerank(pipeline, question, rerank_k=5)
    return docs[0].metadata["rerank_score"] if docs else 0.0


def describe(scores: list[float]) -> str:
    return (
        f"min={min(scores):.4f} median={statistics.median(scores):.4f} "
        f"max={max(scores):.4f}"
    )


def main():
    parser = argparse.ArgumentParser(description="校准精排得分拒答阈值（不调用 LLM）")
    parser.add_argument("--mode", choices=("vector", "bm25", "hybrid"), default="hybrid")
    args = parser.parse_args()

    print("加载本地 embedding 与 reranker 模型...")
    pipeline, _ = build_pipeline(retrieve_k=20, mode=args.mode)

    answerable = []
    print("\n(a) 库内可回答（评测题 + 口语化补充题）：top1 rerank_score")
    for question in load_answerable():
        score = top1_score(pipeline, question)
        answerable.append(score)
        print(f"  {score:.4f}  {question}")

    out_of_scope = []
    print("\n(b) 库外问题：top1 rerank_score")
    for kind, question in OUT_OF_SCOPE:
        score = top1_score(pipeline, question)
        out_of_scope.append(score)
        print(f"  {score:.4f}  [{kind}] {question}")

    print("\n各组统计")
    print(f"  (a) 可回答 {len(answerable)} 题：{describe(answerable)}")
    print(f"  (b) 库外   {len(out_of_scope)} 题：{describe(out_of_scope)}")
    for kind in dict.fromkeys(kind for kind, _ in OUT_OF_SCOPE):
        scores = [s for (k, _), s in zip(OUT_OF_SCOPE, out_of_scope) if k == kind]
        print(f"      {kind}（{len(scores)} 题）：{describe(scores)}")

    print("\n候选阈值（拒答条件：top1 < 阈值）")
    print("  阈值   误拒(a)  漏拦(b)")
    for threshold in CANDIDATE_THRESHOLDS:
        false_refusals = sum(s < threshold for s in answerable)
        false_answers = sum(s >= threshold for s in out_of_scope)
        print(
            f"  {threshold:<5} {false_refusals:>3}/{len(answerable)}"
            f"   {false_answers:>3}/{len(out_of_scope)}"
        )

    lowest = min(answerable)
    recommended = round(lowest * SAFETY_MARGIN, 2)
    caught = sum(s < recommended for s in out_of_scope)
    print(f"\n可回答题最低分 {lowest:.4f}；推荐阈值 = 最低分 × {SAFETY_MARGIN} ≈ {recommended}")
    print(f"  该阈值下：误拒 0/{len(answerable)}，拦下库外问题 {caught}/{len(out_of_scope)}")
    overlap = [
        (score, question)
        for (_, question), score in zip(OUT_OF_SCOPE, out_of_scope)
        if score >= recommended
    ]
    if overlap:
        print("  与可回答题分数重叠、未被拦下的库外问题：")
        for score, question in sorted(overlap, reverse=True):
            print(f"    {score:.4f}  {question}")


if __name__ == "__main__":
    main()
