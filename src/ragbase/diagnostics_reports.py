"""体检报告历史（数据库里的 diagnose_reports 表）：保存 build_report 的结果，读取、列出、删除。

- 报告的几大块（params / summary / recommendations / 逐题记录）各存一列 JSON；extra 存 question_ids、question_digest、
  concurrent_chat_requests、warmup_question。读回来的结构与 build_report 相同（逐题记录仍叫 questions），另加 id 和 origin。
- 保留到手动删除为止。本模块只用标准库，不导入 ragas / 模型相关代码。
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ragbase.db import Database

ORIGINS = ("cli", "web")


def _dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False)


def _dig(data: dict | None, *path: str):
    """沿路径取嵌套值，中途遇到 None / 缺键返回 None。"""
    for key in path:
        if not isinstance(data, dict):
            return None
        data = data.get(key)
    return data


class ReportStore:
    def __init__(self, db: "Database"):
        self._db = db

    def save(self, report: dict, origin: str) -> int:
        """保存一份 build_report 产出的报告，返回新报告的 id。origin 是 "cli" 或 "web"。"""
        if origin not in ORIGINS:
            raise ValueError(f"origin 只能是 {ORIGINS}，收到 {origin!r}")
        extra = {
            "question_ids": report.get("question_ids", []),
            "question_digest": report.get("question_digest"),
            "concurrent_chat_requests": report.get("concurrent_chat_requests", 0),
            "warmup_question": report.get("warmup_question"),
        }
        with self._db.transaction() as conn:
            cursor = conn.execute(
                "INSERT INTO diagnose_reports "
                "(created_at, origin, mode, question_count, params, summary, recommendations, records, extra) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    report["generated_at"], origin, report["params"]["mode"], report["summary"]["questions"]["measured"],
                    _dumps(report["params"]), _dumps(report["summary"]), _dumps(report["recommendations"]),
                    _dumps(report["questions"]), _dumps(extra),
                ),
            )
            return cursor.lastrowid

    def list(self) -> list[dict]:
        """报告列表（不含逐题记录），按 id 倒序；每项带几个关键数字，没有对应数据时为 None。"""
        rows = self._db.query(
            "SELECT id, created_at, origin, mode, question_count, summary, extra FROM diagnose_reports ORDER BY id DESC"
        )
        items = []
        for row in rows:
            summary, extra = json.loads(row["summary"]), json.loads(row["extra"])
            items.append({
                "id": row["id"],
                "created_at": row["created_at"],
                "origin": row["origin"],
                "mode": row["mode"],
                "question_count": row["question_count"],
                "candidates_recall": _dig(summary, "quality", "candidates", "point_recall"),
                "reranked_recall": _dig(summary, "quality", "reranked", "point_recall"),
                "refusal_rate": _dig(summary, "quality", "refusal_rate"),
                "total_ms_mean": _dig(summary, "speed", "total", "mean"),
                "rerank_share_pct": _dig(summary, "speed", "stages", "rerank", "share_pct"),
                "concurrent_chat_requests": extra.get("concurrent_chat_requests", 0),
            })
        return items

    def get(self, report_id: int) -> dict | None:
        rows = self._db.query("SELECT * FROM diagnose_reports WHERE id = ?", (report_id,))
        if not rows:
            return None
        row = rows[0]
        extra = json.loads(row["extra"])
        return {
            "id": row["id"],
            "origin": row["origin"],
            "generated_at": row["created_at"],
            "params": json.loads(row["params"]),
            "warmup_question": extra.get("warmup_question"),
            "questions": json.loads(row["records"]),
            "summary": json.loads(row["summary"]),
            "recommendations": json.loads(row["recommendations"]),
            "question_ids": extra.get("question_ids", []),
            "question_digest": extra.get("question_digest"),
            "concurrent_chat_requests": extra.get("concurrent_chat_requests", 0),
        }

    def delete(self, report_id: int) -> bool:
        with self._db.transaction() as conn:
            return conn.execute("DELETE FROM diagnose_reports WHERE id = ?", (report_id,)).rowcount > 0
