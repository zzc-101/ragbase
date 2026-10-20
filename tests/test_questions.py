"""评测题集（QuestionStore）与体检报告历史（ReportStore）的单元测试，以及 CLI 的 questions 子命令。

全部使用临时目录里的数据库文件（RAG_HOME / RAG_DB_PATH 指向临时目录）：不碰项目里真实的 ragbase.db，
不加载模型，不调用大模型。项目根目录的 eval_questions.json 只读，不修改。

运行：uv run python tests/test_questions.py
"""
import contextlib
import io
import json
import os
import tempfile
from pathlib import Path

from ragbase import cli
from ragbase.config import get_settings
from ragbase.db import Database, open_database
from ragbase.diagnostics_reports import ReportStore
from ragbase.evaluation.questions import (
    IMPORTED_META_KEY,
    QuestionStore,
    QuestionValidationError,
    ensure_initial_import,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SCRATCH = tempfile.TemporaryDirectory(prefix="ragdemo-questions-test-")
SCRATCH = Path(_SCRATCH.name).resolve()
_counter = iter(range(10_000))


def _fresh_dir(name: str = "home") -> Path:
    path = SCRATCH / f"{name}-{next(_counter)}"
    path.mkdir()
    return path


@contextlib.contextmanager
def env(**values):
    """临时设置环境变量，并先清掉所有 RAG_* / ARK_* 变量，保证与外部环境无关。"""
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


def _store() -> tuple[Database, QuestionStore]:
    db = Database(":memory:")
    return db, QuestionStore(db)


def _errors(store: QuestionStore, items, mode="append") -> list[dict]:
    try:
        store.import_items(items, mode)
    except QuestionValidationError as exc:
        return exc.errors
    raise AssertionError("应抛出 QuestionValidationError")


# ---------------------------------------------------------------- 增删改查

def test_crud():
    db, store = _store()
    first = store.create({"question": "什么是软件工程？", "reference": "工程化方法，提高质量", "difficulty": "简单"})
    second = store.create({"question": "什么是软件架构？", "type": "概念题", "id": 999})  # create 忽略传入的 id
    assert (first["id"], second["id"]) == (1, 2)
    assert first["reference"] == "工程化方法，提高质量" and first["type"] is None and second["reference"] is None
    assert first["created_at"] == first["updated_at"] and "T" in first["created_at"]  # ISO 时间
    assert store.get(1) == first and store.get(99) is None
    assert [q["id"] for q in store.list()] == [1, 2]

    updated = store.update(1, {"difficulty": "困难", "reference": None})  # 只改传入的字段，null 清空
    assert updated["difficulty"] == "困难" and updated["reference"] is None
    assert updated["question"] == "什么是软件工程？" and updated["created_at"] == first["created_at"]
    assert updated["updated_at"] >= first["updated_at"]
    assert store.update(1, {})["question"] == "什么是软件工程？"  # 空更新：原样返回
    assert store.update(99, {"question": "不存在"}) is None
    assert store.update(2, {"answer_keyword": "旧字段名也认"})["reference"] == "旧字段名也认"

    assert store.delete(1) is True and store.delete(1) is False
    assert [q["id"] for q in store.list()] == [2]
    assert store.create({"question": "新题"})["id"] == 3  # 最大 id + 1


def test_create_and_update_validate_like_import():
    db, store = _store()
    for bad in ({}, {"question": "  "}, {"question": 5}, {"question": "q", "difficulty": 3}, {"question": "q", "reference": ["a"]}):
        try:
            store.create(bad)
        except QuestionValidationError as exc:
            assert exc.errors and all(e["index"] is None and e["message"] for e in exc.errors), bad
        else:
            raise AssertionError(f"{bad!r} 应校验失败")
    created = store.create({"question": "q"})
    for bad in ({"question": ""}, {"type": 1}, {"source_topic": {}}):
        try:
            store.update(created["id"], bad)
        except QuestionValidationError:
            pass
        else:
            raise AssertionError(f"{bad!r} 应校验失败")
    assert store.get(created["id"])["question"] == "q" and len(store.list()) == 1  # 失败不写入
    try:
        store.create([1, 2])
    except QuestionValidationError as exc:
        assert "对象" in exc.errors[0]["message"]
    else:
        raise AssertionError("非对象应校验失败")


# ---------------------------------------------------------------- 导入校验与原子性

def test_import_validation_is_atomic():
    db, store = _store()
    store.create({"question": "已有题", "reference": "保持原样"})
    before = store.list()

    errors = _errors(store, {"question": "不是数组"})  # 整体不是数组：一条整体错误，index 为 None
    assert len(errors) == 1 and errors[0]["index"] is None and "数组" in errors[0]["message"]

    errors = _errors(store, [{"question": "好题"}, {"reference": "缺 question"}, {"question": "  "}])
    assert [(e["index"], e["field"]) for e in errors] == [(1, "question"), (2, "question")]
    assert all(isinstance(e["message"], str) and e["message"] for e in errors)

    errors = _errors(store, [{"id": "7", "question": "id 是字符串"}, {"id": 0, "question": "q"}, {"id": -1, "question": "q"},
                             {"id": True, "question": "q"}, {"id": 1.5, "question": "q"}])
    assert [(e["index"], e["field"]) for e in errors] == [(i, "id") for i in range(5)]

    errors = _errors(store, [{"id": 5, "question": "a"}, {"id": 5, "question": "b"}])  # 同一批里 id 重复
    assert [(e["index"], e["field"]) for e in errors] == [(1, "id")]

    errors = _errors(store, [{"question": "q", "difficulty": 1}, {"question": "q", "reference": 2}, {"question": "q", "answer_keyword": []},
                             {"question": "q", "type": {}}, {"question": "q", "source_topic": False}, "不是对象", None])
    assert [(e["index"], e["field"]) for e in errors] == [
        (0, "difficulty"), (1, "reference"), (2, "answer_keyword"), (3, "type"), (4, "source_topic"), (5, None), (6, None)]

    # 好题与坏题同批：整体不写入，包括 replace 模式下“先清空”也被回滚
    for mode in ("append", "replace"):
        _errors(store, [{"question": "好题"}, {"question": 1}], mode)
        assert store.list() == before, mode
    try:
        store.import_items([], "merge")
    except ValueError:
        pass
    else:
        raise AssertionError("未知 mode 应报错")


def test_optional_fields_accept_null_and_strings():
    db, store = _store()
    result = store.import_items([{"question": "q1", "reference": None, "difficulty": None, "type": "概念题", "source_topic": "主题"}], "append")
    assert result == {"imported": 1, "reassigned": []}
    assert store.get(1)["type"] == "概念题" and store.get(1)["reference"] is None


def test_import_append_reassigns_conflicting_ids():
    db, store = _store()
    store.import_items([{"id": 1, "question": "旧一"}, {"id": 5, "question": "旧五"}], "append")
    result = store.import_items([
        {"id": 5, "question": "新五（冲突）"},
        {"id": 2, "question": "新二（不冲突）"},
        {"question": "没写 id"},
        {"id": 1, "question": "新一（冲突）"},
    ], "append")
    assert result["imported"] == 4
    # 冲突的题改用新 id（排在已有 id 与文件里出现的 id 之后）；没冲突的保留，没写 id 的自动分配
    assert result["reassigned"] == [[5, 6], [1, 8]], result
    by_question = {q["question"]: q["id"] for q in store.list()}
    assert by_question == {"旧一": 1, "旧五": 5, "新五（冲突）": 6, "新二（不冲突）": 2, "没写 id": 7, "新一（冲突）": 8}
    assert store.get(5)["question"] == "旧五"  # 已有题目未被覆盖


def test_import_replace_keeps_file_ids():
    db, store = _store()
    store.import_items([{"id": 1, "question": "旧一"}, {"id": 2, "question": "旧二"}], "append")
    result = store.import_items([{"id": 10, "question": "新十"}, {"question": "没写 id"}, {"id": 2, "question": "新二"}], "replace")
    assert result == {"imported": 3, "reassigned": []}
    assert {q["question"]: q["id"] for q in store.list()} == {"新十": 10, "没写 id": 11, "新二": 2}
    assert store.import_items([], "replace") == {"imported": 0, "reassigned": []}
    assert store.list() == []


# ---------------------------------------------------------------- reference / 导出兼容

def test_reference_falls_back_to_answer_keyword():
    db, store = _store()
    store.import_items([
        {"question": "只有旧字段", "answer_keyword": "旧要点一，旧要点二"},
        {"question": "两者都有", "reference": "新参考答案", "answer_keyword": "旧要点"},
        {"question": "reference 为空串时取旧字段", "reference": "", "answer_keyword": "回退值"},
        {"question": "都没有"},
    ], "append")
    references = {q["question"]: q["reference"] for q in store.list()}
    assert references == {"只有旧字段": "旧要点一，旧要点二", "两者都有": "新参考答案", "reference 为空串时取旧字段": "回退值", "都没有": None}


def test_export_is_backward_compatible():
    db, store = _store()
    store.import_items([
        {"id": 3, "question": "有参考", "reference": "要点甲，要点乙", "difficulty": "简单", "type": "概念题", "source_topic": "主题"},
        {"id": 4, "question": "旧格式", "answer_keyword": "旧要点"},
    ], "append")
    exported = json.loads(json.dumps(store.export(), ensure_ascii=False))  # 经过一次 JSON 往返
    for item in exported:
        assert item["reference"] == item["answer_keyword"]
        assert {"id", "question", "reference", "answer_keyword", "difficulty", "type", "source_topic"} <= set(item)
    # 旧版读法：reference or answer_keyword
    assert [item.get("reference") or item.get("answer_keyword") for item in exported] == ["要点甲，要点乙", "旧要点"]
    # 导出再导入得到同样的内容
    other_db, other = _store()
    other.import_items(exported, "replace")
    strip = lambda items: [{k: v for k, v in q.items() if k not in ("created_at", "updated_at")} for q in items]  # noqa: E731
    assert strip(other.list()) == strip(store.list())


# ---------------------------------------------------------------- 首次导入

def test_initial_import_runs_once():
    home = _fresh_dir()
    items = [{"id": 1, "question": "甲", "answer_keyword": "要点甲"}, {"id": 2, "question": "乙", "answer_keyword": "要点乙"}]
    (home / "eval_questions.json").write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    with env(RAG_HOME=str(home), RAG_DB_PATH=str(home / "test.db")):
        db = open_database()
        store = QuestionStore(db)
        assert [q["question"] for q in store.list()] == ["甲", "乙"]
        assert db.meta_get(IMPORTED_META_KEY) is not None
        for question in store.list():
            store.delete(question["id"])  # 用户删光题目
        db.close()

        db = open_database()  # 再次打开：不会重新导入
        assert QuestionStore(db).list() == []
        assert ensure_initial_import(db, home / "eval_questions.json") is None
        db.close()
    assert not (home / "ragbase.db").exists()  # 数据库在 RAG_DB_PATH 指定的位置


def test_initial_import_skipped_when_table_not_empty_or_file_bad():
    home = _fresh_dir()
    path = home / "eval_questions.json"
    path.write_text(json.dumps([{"question": "文件里的题", "answer_keyword": "要点"}], ensure_ascii=False), encoding="utf-8")
    db = Database(":memory:")
    QuestionStore(db).create({"question": "已有题"})
    assert ensure_initial_import(db, path) is None and db.meta_get(IMPORTED_META_KEY) is None  # 表非空：不导入，也不写标记
    db.close()

    for content in ("不是 JSON", '{"question": "不是数组"}', '[{"id": "x", "question": "坏"}]'):
        db = Database(":memory:")
        path.write_text(content, encoding="utf-8")
        assert ensure_initial_import(db, path) is None, content
        assert QuestionStore(db).list() == [] and db.meta_get(IMPORTED_META_KEY) is None  # 不写标记：修好文件后可重试
        path.write_text("[]", encoding="utf-8")
        assert ensure_initial_import(db, path) == 0 and db.meta_get(IMPORTED_META_KEY) is not None
        db.close()


def test_initial_import_missing_file_returns_none():
    db = Database(":memory:")
    assert ensure_initial_import(db, _fresh_dir() / "eval_questions.json") is None
    assert db.meta_get(IMPORTED_META_KEY) is None  # 没有文件时不写标记：之后放入文件还能导入
    db.close()


def test_initial_import_of_project_eval_questions_file():
    source = PROJECT_ROOT / "eval_questions.json"
    original = json.loads(source.read_text(encoding="utf-8"))
    assert len(original) == 30
    before = source.read_bytes()
    db = Database(":memory:")
    assert ensure_initial_import(db, source) == 30
    questions = QuestionStore(db).list()
    assert [q["id"] for q in questions] == [item["id"] for item in original]
    for question, item in zip(questions, original):
        assert question["question"] == item["question"]
        assert question["reference"] == item["answer_keyword"]
        assert (question["difficulty"], question["type"], question["source_topic"]) == (item["difficulty"], item["type"], item["source_topic"])
    db.close()
    assert source.read_bytes() == before  # 只读，没有被修改


# ---------------------------------------------------------------- 报告历史

def _report(mode="hybrid", concurrent=0, **summary_overrides):
    quality = {
        "candidates": {"point_recall": 0.9, "context_precision": 0.5, "mrr": 0.8},
        "reranked": {"point_recall": 0.85, "context_precision": 0.6, "mrr": 0.8},
        "llm_context": {"point_recall": 0.8, "context_precision": 0.7, "mrr": 0.8},
        "refused": 1, "refusal_rate": 0.1,
    }
    speed = {"total": {"mean": 123.4, "median": 120.0, "max": 200.0, "min": 80.0, "count": 2},
             "stages": {"rerank": {"mean": 80.0, "median": 80.0, "max": 100.0, "min": 60.0, "count": 2, "share_pct": 64.8}}}
    summary = {"questions": {"measured": 2, "with_reference": 2, "duplicates_skipped": 0}, "quality": quality, "speed": speed}
    summary.update(summary_overrides)
    return {
        "generated_at": "2026-10-20T10:00:00",
        "params": {"mode": mode, "rerank_k": 5},
        "warmup_question": "预热用的占位问题：请忽略这条",
        "questions": [{"id": 1, "question": "甲"}, {"id": 2, "question": "乙"}],
        "summary": summary,
        "recommendations": [{"id": "rerank_slow", "title": "精排偏慢", "phenomenon": "现象", "try": "尝试", "cost": "代价"}],
        "question_ids": [1, 2],
        "question_digest": "abc123",
        "concurrent_chat_requests": concurrent,
    }


def test_report_store_roundtrip_and_list_fields():
    db = Database(":memory:")
    store = ReportStore(db)
    assert store.list() == [] and store.get(1) is None
    first = _report(mode="hybrid", concurrent=3)
    first_id = store.save(first, "cli")
    second_id = store.save(_report(mode="vector"), "web")
    assert (first_id, second_id) == (1, 2)

    loaded = store.get(first_id)
    assert loaded["id"] == first_id and loaded["origin"] == "cli"
    assert {key: value for key, value in loaded.items() if key not in ("id", "origin")} == first  # 结构与 build_report 相同
    assert store.get(99) is None

    listing = store.list()
    assert [item["id"] for item in listing] == [2, 1]  # id 倒序
    assert set(listing[1]) == {"id", "created_at", "origin", "mode", "question_count", "candidates_recall", "reranked_recall",
                               "refusal_rate", "total_ms_mean", "rerank_share_pct", "concurrent_chat_requests"}
    assert listing[1] == {
        "id": 1, "created_at": "2026-10-20T10:00:00", "origin": "cli", "mode": "hybrid", "question_count": 2,
        "candidates_recall": 0.9, "reranked_recall": 0.85, "refusal_rate": 0.1, "total_ms_mean": 123.4,
        "rerank_share_pct": 64.8, "concurrent_chat_requests": 3,
    }
    assert listing[0]["origin"] == "web" and listing[0]["mode"] == "vector" and listing[0]["concurrent_chat_requests"] == 0

    # 没有带参考答案的题 / 没有精排：对应数字为 None，不报错
    bare = _report()
    bare["summary"]["quality"].update(candidates=None, reranked=None, llm_context=None)
    bare["summary"]["speed"]["stages"] = {}
    store.save(bare, "cli")
    newest = store.list()[0]
    assert newest["candidates_recall"] is None and newest["reranked_recall"] is None and newest["rerank_share_pct"] is None

    assert store.delete(first_id) is True and store.delete(first_id) is False
    assert [item["id"] for item in store.list()] == [3, 2]
    assert store.save(_report(), "cli") == 4  # AUTOINCREMENT：删除后 id 不复用
    try:
        store.save(_report(), "api")
    except ValueError:
        pass
    else:
        raise AssertionError("未知 origin 应报错")
    db.close()


# ---------------------------------------------------------------- CLI

def _run_cli(*argv) -> tuple[int, str]:
    buffer = io.StringIO()
    code = 0
    with contextlib.redirect_stdout(buffer):
        try:
            args = cli.build_parser().parse_args(list(argv))
            args.func(args)
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 1
    return code, buffer.getvalue()


def test_cli_questions_import_and_export():
    home = _fresh_dir()
    with env(RAG_HOME=str(home), RAG_DB_PATH=str(home / "cli.db")):
        good = home / "good.json"
        good.write_text(json.dumps([{"id": 1, "question": "甲", "answer_keyword": "要点甲"}, {"id": 2, "question": "乙"}], ensure_ascii=False), encoding="utf-8")
        code, out = _run_cli("questions", "import", str(good))
        assert code == 0 and "已导入 2 道题" in out, out
        code, out = _run_cli("questions", "import", str(good))  # 再追加一次：id 冲突，换新 id
        assert code == 0 and "已改用新 id" in out and "1 → 3" in out and "2 → 4" in out, out
        code, out = _run_cli("questions", "import", str(good), "--replace")
        assert code == 0 and "替换全部" in out and "已改用新 id" not in out, out

        exported = home / "out.json"
        code, out = _run_cli("questions", "export", str(exported))
        assert code == 0 and "已导出 2 道题" in out, out
        data = json.loads(exported.read_text(encoding="utf-8"))
        assert [(item["id"], item["reference"], item["answer_keyword"]) for item in data] == [(1, "要点甲", "要点甲"), (2, None, None)]

        bad = home / "bad.json"
        bad.write_text(json.dumps([{"question": "好"}, {"id": "x", "question": ""}]), encoding="utf-8")
        code, out = _run_cli("questions", "import", str(bad), "--replace")
        assert code != 0 and "第 2 题" in out and "question" in out and "id" in out, out
        bad.write_text('{"不是": "数组"}', encoding="utf-8")
        code, out = _run_cli("questions", "import", str(bad))
        assert code != 0 and "数组" in out, out
        bad.write_text("不是 JSON", encoding="utf-8")
        code, _ = _run_cli("questions", "import", str(bad))
        assert code != 0
        code, _ = _run_cli("questions", "import", str(home / "missing.json"))
        assert code != 0

        db = open_database()  # 校验失败的导入没有改动题集
        assert [q["question"] for q in QuestionStore(db).list()] == ["甲", "乙"]
        db.close()
    assert not (home / "ragbase.db").exists()


def test_cli_registers_questions_subcommands():
    args = cli.build_parser().parse_args(["questions", "import", "x.json", "--replace"])
    assert args.command == "questions" and args.file == "x.json" and args.replace is True and args.func is cli._cmd_questions_import
    args = cli.build_parser().parse_args(["questions", "export", "y.json"])
    assert args.file == "y.json" and args.func is cli._cmd_questions_export
    assert "questions" in cli.build_parser().format_usage()


def test_cli_reports_database_error_in_chinese():
    home = _fresh_dir()
    blocker = home / "not-a-dir"
    blocker.write_text("x", encoding="utf-8")  # 数据库目录的父路径是个文件：建不了目录
    with env(RAG_HOME=str(home), RAG_DB_PATH=str(blocker / "sub" / "x.db")):
        code, _ = _run_cli("questions", "export", str(home / "out.json"))
        assert code != 0 and not (home / "out.json").exists()
        args = cli.build_parser().parse_args(["questions", "export", str(home / "out.json")])
        try:
            args.func(args)
        except SystemExit as exc:
            assert "数据库" in str(exc.code), exc.code
        else:
            raise AssertionError("应以 SystemExit 退出")


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"通过：{test.__name__}")
    print(f"共 {len(tests)} 项测试通过")
