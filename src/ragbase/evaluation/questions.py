"""评测题集（数据库里的 eval_questions 表）：增删改查、校验、导入 / 导出、首次从 eval_questions.json 导入。

- 本模块很轻（只用标准库），不导入 ragas；体检（diagnostics）、CLI、HTTP API 和评测（runner）共用它。
- 题目字段：id、question、reference、difficulty、type、source_topic、created_at、updated_at。
  `reference` 是参考答案，可空：空时这题只计耗时、不计召回率（ragbase eval 会跳过它）。
- 导入的 JSON 格式与历史上的 eval_questions.json 一致：`reference` 取 `reference`，没有则取 `answer_keyword`；
  导出时 `reference` 与 `answer_keyword` 写同一个值，所以导出的文件旧版 `ragbase eval` 也能读。
- 校验失败一律抛 QuestionValidationError，`errors` 里每项是 {index, field, message}：index 是该题在导入数组里的下标（从 0 开始；
  不针对某一题的错误，如“输入不是数组”，以及 create / update 的错误，index 为 None），message 是中文。
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from ragbase.db import Database

logger = logging.getLogger(__name__)

IMPORTED_META_KEY = "eval_questions_imported"  # meta 表里“已做过首次导入”的标记
OPTIONAL_FIELDS = ("reference", "difficulty", "type", "source_topic")
COLUMNS = ("id", "question", "reference", "difficulty", "type", "source_topic", "created_at", "updated_at")


class QuestionValidationError(ValueError):
    """题目校验失败；errors 是 [{index, field, message}, …]，message 为中文。"""

    def __init__(self, errors: list[dict]):
        self.errors = errors
        super().__init__("；".join(_describe(error) for error in errors) or "题目校验失败")


def _describe(error: dict) -> str:
    where = f"第 {error['index'] + 1} 题" if error.get("index") is not None else ""
    field = f"（{error['field']}）" if error.get("field") else ""
    return f"{where}{field}{error['message']}"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _type_name(value: object) -> str:
    return type(value).__name__


def _clean_text(value: str) -> str | None:
    value = value.strip()
    return value or None


def _validate_fields(data: dict, index: int | None, partial: bool) -> tuple[dict, list[dict]]:
    """校验并规整题目字段（导入、create、update 共用）。返回 (干净字段, 错误列表)。

    partial=False（导入 / create）：question 必填；reference 取 reference，没有则取 answer_keyword。
    partial=True（update）：只处理传入的字段，没传的不出现在结果里；传了 null 表示清空该可选字段。
    """
    errors: list[dict] = []
    clean: dict = {}

    def error(field: str, message: str) -> None:
        errors.append({"index": index, "field": field, "message": message})

    if "question" in data or not partial:
        value = data.get("question")
        if not isinstance(value, str) or not value.strip():
            error("question", "必须是非空字符串" if "question" in data else "缺少必填字段 question（非空字符串）")
        else:
            clean["question"] = value.strip()

    texts: dict[str, str | None] = {}
    for name in (*OPTIONAL_FIELDS, "answer_keyword"):
        if name not in data:
            continue
        value = data[name]
        if value is not None and not isinstance(value, str):
            error(name, f"必须是字符串或 null（实际是 {_type_name(value)}）")
            continue
        texts[name] = _clean_text(value) if isinstance(value, str) else None

    for name in ("difficulty", "type", "source_topic"):
        if name in texts:
            clean[name] = texts[name]
    if partial:
        if "reference" in texts:
            clean["reference"] = texts["reference"]
        elif "answer_keyword" in texts:
            clean["reference"] = texts["answer_keyword"]
    else:
        clean["reference"] = texts.get("reference") or texts.get("answer_keyword")
        for name in ("difficulty", "type", "source_topic"):
            clean.setdefault(name, None)
    return clean, errors


def _validate_id(data: dict, index: int | None) -> tuple[int | None, list[dict]]:
    """id 可以不写（或写 null）；写了必须是正整数（布尔值不算）。"""
    value = data.get("id")
    if value is None:
        return None, []
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return None, [{"index": index, "field": "id", "message": f"必须是正整数（实际是 {value!r}）"}]
    return value, []


def _row_dict(row: sqlite3.Row) -> dict:
    return {name: row[name] for name in COLUMNS}


def _import_in(conn: sqlite3.Connection, items: object, mode: str) -> dict:
    """在调用方已开启的事务里校验并写入题目；校验失败抛 QuestionValidationError，此时尚未写入任何数据。"""
    if mode not in ("append", "replace"):
        raise ValueError(f"mode 只能是 append 或 replace，收到 {mode!r}")
    if not isinstance(items, list):
        raise QuestionValidationError([{
            "index": None, "field": None,
            "message": f"题集应是题目组成的 JSON 数组（实际是 {_type_name(items)}）",
        }])

    errors: list[dict] = []
    parsed: list[tuple[int | None, dict]] = []
    first_seen: dict[int, int] = {}
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            errors.append({"index": index, "field": None, "message": f"应是 JSON 对象（实际是 {_type_name(item)}）"})
            continue
        fields, field_errors = _validate_fields(item, index, partial=False)
        item_id, id_errors = _validate_id(item, index)
        errors += field_errors + id_errors
        if item_id is not None and not id_errors:
            if item_id in first_seen:
                errors.append({
                    "index": index, "field": "id",
                    "message": f"id {item_id} 重复（与第 {first_seen[item_id] + 1} 题相同）",
                })
            else:
                first_seen[item_id] = index
        parsed.append((item_id, fields))
    if errors:
        raise QuestionValidationError(errors)

    if mode == "replace":
        conn.execute("DELETE FROM eval_questions")
        existing: set[int] = set()
    else:
        existing = {row[0] for row in conn.execute("SELECT id FROM eval_questions")}
    # 新分配的 id 一律排在“已有 id 与文件里出现的 id”之后，不会与任何一个冲突
    next_id = max(existing | set(first_seen), default=0) + 1
    now = _now()
    reassigned: list[list[int]] = []
    for item_id, fields in parsed:
        if item_id is None or item_id in existing:
            new_id, next_id = next_id, next_id + 1
            if item_id is not None:
                reassigned.append([item_id, new_id])
            item_id = new_id
        conn.execute(
            "INSERT INTO eval_questions (id, question, reference, difficulty, type, source_topic, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (item_id, fields["question"], fields["reference"], fields["difficulty"], fields["type"],
             fields["source_topic"], now, now),
        )
    return {"imported": len(parsed), "reassigned": reassigned}


class QuestionStore:
    """评测题集的读写（单个题集）。写操作各自一个事务；多线程可共用同一个 Database。"""

    def __init__(self, db: "Database"):
        self._db = db

    def list(self) -> list[dict]:
        return [_row_dict(row) for row in self._db.query(f"SELECT {', '.join(COLUMNS)} FROM eval_questions ORDER BY id")]

    def get(self, question_id: int) -> dict | None:
        rows = self._db.query(f"SELECT {', '.join(COLUMNS)} FROM eval_questions WHERE id = ?", (question_id,))
        return _row_dict(rows[0]) if rows else None

    def create(self, data: dict) -> dict:
        """新增一道题，id 自动分配（data 里的 id 被忽略）；字段校验失败抛 QuestionValidationError。"""
        if not isinstance(data, dict):
            raise QuestionValidationError([{"index": None, "field": None, "message": f"应是 JSON 对象（实际是 {_type_name(data)}）"}])
        fields, errors = _validate_fields(data, None, partial=False)
        if errors:
            raise QuestionValidationError(errors)
        now = _now()
        with self._db.transaction() as conn:
            new_id = conn.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM eval_questions").fetchone()[0]
            conn.execute(
                "INSERT INTO eval_questions (id, question, reference, difficulty, type, source_topic, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (new_id, fields["question"], fields["reference"], fields["difficulty"], fields["type"],
                 fields["source_topic"], now, now),
            )
        return {"id": new_id, **fields, "created_at": now, "updated_at": now}

    def update(self, question_id: int, data: dict) -> dict | None:
        """只改传入的字段（传 null 清空可选字段）；题目不存在返回 None，校验失败抛 QuestionValidationError。"""
        if not isinstance(data, dict):
            raise QuestionValidationError([{"index": None, "field": None, "message": f"应是 JSON 对象（实际是 {_type_name(data)}）"}])
        fields, errors = _validate_fields(data, None, partial=True)
        if errors:
            raise QuestionValidationError(errors)
        with self._db.transaction() as conn:
            if conn.execute("SELECT 1 FROM eval_questions WHERE id = ?", (question_id,)).fetchone() is None:
                return None
            if fields:
                assignments = ", ".join(f"{name} = ?" for name in fields)
                conn.execute(
                    f"UPDATE eval_questions SET {assignments}, updated_at = ? WHERE id = ?",
                    (*fields.values(), _now(), question_id),
                )
        return self.get(question_id)

    def delete(self, question_id: int) -> bool:
        with self._db.transaction() as conn:
            return conn.execute("DELETE FROM eval_questions WHERE id = ?", (question_id,)).rowcount > 0

    def import_items(self, items: object, mode: Literal["append", "replace"]) -> dict:
        """批量导入。返回 {"imported": 条数, "reassigned": [[旧 id, 新 id], …]}。

        append：已存在的 id 冲突时改用新 id（记在 reassigned 里）；replace：先清空，再按文件里的 id 写入。
        没写 id 的题自动分配。校验与写入在同一个事务里：校验失败抛 QuestionValidationError，库保持原样。
        """
        with self._db.transaction() as conn:
            return _import_in(conn, items, mode)

    def export(self) -> list[dict]:
        """导出为与 eval_questions.json 同格式的列表：reference 与 answer_keyword 写同一个值，旧版 eval 也能读。"""
        return [
            {
                "id": q["id"], "question": q["question"], "reference": q["reference"], "answer_keyword": q["reference"],
                "source_topic": q["source_topic"], "difficulty": q["difficulty"], "type": q["type"],
            }
            for q in self.list()
        ]


def ensure_initial_import(db: "Database", json_path: Path) -> int | None:
    """首次打开数据库时，把 eval_questions.json 导入题集表（只做一次）。返回导入条数；没有导入返回 None。

    只有“没有导入标记、题集表为空、文件存在”才导入，导入与写标记在同一个事务里；之后即使用户删光题目也不会再导入。
    文件读不出来 / 校验不通过时只记一条警告并返回 None（不写标记，修好文件后下次打开会重试），不让它挡住其它命令。
    """
    if db.meta_get(IMPORTED_META_KEY) is not None or not json_path.is_file():
        return None
    try:
        items = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("无法读取评测题集 %s，跳过首次导入：%s", json_path, exc)
        return None
    with db.transaction() as conn:
        # 拿到写锁后再确认一次：别的进程可能刚导入过
        if conn.execute("SELECT 1 FROM meta WHERE key = ?", (IMPORTED_META_KEY,)).fetchone() is not None:
            return None
        if conn.execute("SELECT 1 FROM eval_questions LIMIT 1").fetchone() is not None:
            return None
        try:
            result = _import_in(conn, items, "replace")
        except QuestionValidationError as exc:
            logger.warning("评测题集 %s 校验未通过，跳过首次导入：%s", json_path, exc)
            return None
        db.meta_set(conn, IMPORTED_META_KEY, _now())
    return result["imported"]
