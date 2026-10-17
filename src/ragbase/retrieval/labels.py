"""引用来源的文字标签（命令行问答 / 对话用）：召回路径、文件名、页码、章节与精排得分。"""
import os


def route_label(metadata: dict) -> str:
    """召回来源：V=向量、B=BM25，数字为该路内排名，如 V3 B1。"""
    parts = []
    if metadata.get("vec_rank"):
        parts.append(f"V{metadata['vec_rank']}")
    if metadata.get("bm25_rank"):
        parts.append(f"B{metadata['bm25_rank']}")
    return "+".join(parts) or "?"


def source_label(metadata: dict) -> str:
    """引用来源：文件名[:第N页][「末级章节」][召回路径[|精排得分]]；仅在元数据有 page（PDF）时显示页码，有 heading_path 时显示章节。"""
    name = os.path.basename(metadata.get("source", "?"))
    page = metadata.get("page")
    page_text = f":第{page + 1}页" if isinstance(page, int) else ""
    heading = metadata.get("heading_path")
    heading_text = f"「{heading.split(' > ')[-1]}」" if heading else ""  # 只显示最末一级，保持简短
    score = metadata.get("rerank_score")
    score_text = f"|{score:.2f}" if isinstance(score, float) else ""
    return f"{name}{page_text}{heading_text}[{route_label(metadata)}{score_text}]"
