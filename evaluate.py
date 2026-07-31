"""评估脚本：跑 eval_questions.json 里的问题，输出检索与回答质量报告。

运行：uv run evaluate.py

评估两个维度：
1. 检索质量：有没有召回到相关片段（用 answer_keyword 在召回结果里命中判断）
2. 回答质量：用 LLM 当裁判，判断答案是否正确且忠实于文档
"""
import json
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from rag_pipeline import ask, build_pipeline

load_dotenv()

BASE_DIR = Path(__file__).parent
EVAL_FILE = BASE_DIR / "eval_questions.json"

# 检索召回量（评估时可以看更全的召回情况）
RETRIEVE_K = 20
RERANK_K = 5


def load_eval_questions():
    return json.loads(EVAL_FILE.read_text())


def judge_answer(llm, question, answer, keyword):
    """用 LLM 当裁判：答案是否正确、是否提到关键词要点。返回 '正确'/'部分正确'/'错误'。"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是评测裁判。根据问题和参考要点，判断被评测答案的质量。

参考要点（答案应包含这些核心内容）：{keyword}

被评测答案：{answer}

判断标准：
- 答案正确且包含参考要点的核心内容 -> 回答"正确"
- 答案方向对但要点不全或有偏差 -> 回答"部分正确"
- 答案错误、答非所问、或说没找到（但其实文档有） -> 回答"错误"

只回答"正确"、"部分正确"或"错误"三个词之一，不要其它内容。"""),
        ("human", "问题：{question}"),
    ])
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({
        "keyword": keyword, "answer": answer, "question": question,
    }).strip()
    # 兜底：只取前几个字判断
    if "正确" in result and "部分" not in result:
        return "正确"
    elif "部分" in result:
        return "部分正确"
    else:
        return "错误"


def check_retrieval(llm, question, candidates, keyword):
    """用 LLM 判断召回的候选片段是否足以回答该问题（检索质量指标）。

    返回 "充分"/"部分"/"不足"，分别对应召回的相关信息完整/部分/缺失。
    """
    context = "\n\n".join(d.page_content for d in candidates)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是检索质量评估员。判断下面这些召回的文档片段是否足以回答用户的问题。

参考答案要点（问题应该涉及这些内容）：{keyword}

召回的文档片段：
{context}

判断标准：
- 片段包含回答问题所需的完整信息 -> 回答"充分"
- 片段包含部分相关信息，但不完整 -> 回答"部分"
- 片段几乎没有相关信息 -> 回答"不足"

只回答"充分"、"部分"或"不足"三个词之一。"""),
        ("human", "问题：{question}"),
    ])
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({
        "keyword": keyword, "context": context, "question": question,
    }).strip()
    if "充分" in result:
        return "充分"
    elif "部分" in result:
        return "部分"
    else:
        return "不足"


def main():
    questions = load_eval_questions()
    print(f"加载 {len(questions)} 道测试题，开始评估...\n")
    print(f"{'ID':<4}{'难度':<6}{'类型':<10}{'召回率':<8}{'回答':<10}")
    print("-" * 50)

    pipeline, llm = build_pipeline(retrieve_k=RETRIEVE_K)

    results = []
    correct = partial = wrong = 0
    retrieval_full = retrieval_part = retrieval_none = 0

    for q in questions:
        answer, docs, candidates = ask(pipeline, q["question"], rerank_k=RERANK_K)
        verdict = judge_answer(llm, q["question"], answer, q["answer_keyword"])
        retrieval = check_retrieval(llm, q["question"], candidates, q["answer_keyword"])

        if verdict == "正确":
            correct += 1
        elif verdict == "部分正确":
            partial += 1
        else:
            wrong += 1

        if retrieval == "充分":
            retrieval_full += 1
        elif retrieval == "部分":
            retrieval_part += 1
        else:
            retrieval_none += 1

        print(f"{q['id']:<4}{q['difficulty']:<6}{q['type']:<10}检索:{retrieval:<5}回答:{verdict}")
        results.append({
            "id": q["id"],
            "question": q["question"],
            "answer": answer,
            "verdict": verdict,
            "retrieval": retrieval,
        })

    n = len(questions)
    print("\n" + "=" * 50)
    print("评估总结")
    print("=" * 50)
    print(f"回答质量：正确 {correct}/{n}（{correct/n:.0%}） | 部分正确 {partial} | 错误 {wrong}")
    print(f"检索质量：充分 {retrieval_full} | 部分 {retrieval_part} | 不足 {retrieval_none}（充分率 {retrieval_full/n:.0%}）")

    # 保存详细结果
    out = BASE_DIR / "eval_results.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\n详细结果已保存到 {out}")


if __name__ == "__main__":
    main()
