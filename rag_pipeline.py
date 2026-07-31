"""查询与评测共用的 RAG 流水线。"""
import os
from dataclasses import dataclass

import chromadb
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from embed import Qwen3Embeddings
from rag_common import COLLECTION, DB_DIR
from reranker import Qwen3Reranker


ANSWER_SYSTEM_PROMPT = """你是一个文档问答助手。请根据下面提供的上下文片段回答用户的问题。

回答规则：
1. 优先综合多个片段的信息来回答，可以合理推理和归纳，不必局限于单一片段的字面表述。
2. 如果片段中分散地提到了问题相关的不同方面，请把它们整合起来对比说明。
3. 只有当所有片段都完全不相关时，才回答“我没有在文档中找到相关内容”。
4. 回答尽量简洁、准确，并在末尾用 [片段N] 标注引用。

上下文片段：
{context}"""


@dataclass
class RAGPipeline:
    retriever: object
    reranker: Qwen3Reranker
    answer_chain: object


def build_answer_chain(llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system", ANSWER_SYSTEM_PROMPT),
        ("human", "{input}"),
    ])
    return (
        {"context": lambda x: x["context"], "input": lambda x: x["input"]}
        | prompt
        | llm
        | StrOutputParser()
    )


def build_pipeline(retrieve_k: int = 20) -> tuple[RAGPipeline, ChatOpenAI]:
    """加载向量检索、精排和生成模型。"""
    embeddings = Qwen3Embeddings()
    client = chromadb.PersistentClient(path=DB_DIR)
    vectordb = Chroma(
        client=client,
        collection_name=COLLECTION,
        embedding_function=embeddings,
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": retrieve_k})
    reranker = Qwen3Reranker()
    llm = ChatOpenAI(
        model=os.getenv("ARK_CHAT_MODEL"),
        api_key=os.getenv("ARK_CHAT_API_KEY"),
        base_url=os.getenv("ARK_BASE_URL"),
        temperature=0,
    )
    return RAGPipeline(retriever, reranker, build_answer_chain(llm)), llm


def format_docs(docs) -> str:
    return "\n\n".join(
        f"[片段{i + 1}] {doc.page_content}"
        for i, doc in enumerate(docs)
    )


def ask(pipeline: RAGPipeline, question: str, rerank_k: int = 5):
    """执行召回、精排和生成，并同时返回精排片段及原始候选片段。"""
    candidates = pipeline.retriever.invoke(question)
    top_indices = pipeline.reranker.rerank(
        question,
        [doc.page_content for doc in candidates],
        top_k=rerank_k,
    )
    docs = [candidates[index] for index in top_indices]
    answer = pipeline.answer_chain.invoke({
        "context": format_docs(docs),
        "input": question,
    })
    return answer, docs, candidates
