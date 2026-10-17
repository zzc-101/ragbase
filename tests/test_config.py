"""config 的纯单元测试（不加载模型）。

运行：uv run python tests/test_config.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CONTEXT_MIN_SCORE, RERANK_MIN_SCORE, Settings, resolve_device
import rag_pipeline


def _settings(**kwargs):
    return Settings(_env_file=None, **kwargs)


def test_defaults_match_current_behavior():
    settings = _settings(chat_model=None, chat_api_key=None, base_url=None)
    assert settings.device == "auto" and settings.mode == "hybrid"
    assert (settings.retrieve_k, settings.per_route_k, settings.rerank_k) == (20, 30, 5)
    assert settings.min_score == RERANK_MIN_SCORE == 0.7
    assert settings.context_min_score == CONTEXT_MIN_SCORE == 0.95
    assert (settings.host, settings.port, settings.log_level) == ("127.0.0.1", 8000, "INFO")
    assert not settings.llm_configured
    # 数据 / 索引路径默认指向仓库内原有位置，上传上限 300 MB
    root = Path(__file__).resolve().parent.parent
    assert settings.data_dir == root / "data" and settings.db_dir == root / "chroma_db"
    assert settings.meta_file == root / ".ingest_meta.json" and settings.cache_dir == root / ".bm25_cache"
    assert settings.max_upload_mb == 300
    assert rag_pipeline.RERANK_MIN_SCORE == RERANK_MIN_SCORE  # 旧的导入路径仍可用


def test_rag_env_prefix_and_ark_aliases():
    import os
    env = {
        "RAG_PORT": "9001", "RAG_MODE": "bm25", "RAG_LOG_LEVEL": "debug", "RAG_MIN_SCORE": "0",
        "RAG_DATA_DIR": "/tmp/ragdemo-test-data", "RAG_DB_DIR": "/tmp/ragdemo-test-db", "RAG_MAX_UPLOAD_MB": "50",
        "ARK_CHAT_MODEL": "ep-test", "ARK_CHAT_API_KEY": "sk-unit-test", "ARK_BASE_URL": "http://llm.test/v3",
    }
    saved = {key: os.environ.get(key) for key in env}
    os.environ.update(env)
    try:
        settings = _settings()
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    assert settings.port == 9001 and settings.mode == "bm25" and settings.log_level == "DEBUG"
    assert settings.min_score == 0
    assert settings.data_dir == Path("/tmp/ragdemo-test-data") and settings.db_dir == Path("/tmp/ragdemo-test-db")
    assert settings.max_upload_mb == 50
    assert settings.chat_model == "ep-test" and settings.base_url == "http://llm.test/v3"
    assert settings.chat_api_key.get_secret_value() == "sk-unit-test"
    assert "sk-unit-test" not in repr(settings)
    assert settings.llm_configured


def test_invalid_values_rejected():
    for kwargs in ({"mode": "nope"}, {"device": "tpu"}, {"min_score": 2}, {"port": 0}, {"log_level": "loud"}, {"max_upload_mb": 0}):
        try:
            _settings(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"应拒绝 {kwargs}")


def test_resolve_device():
    assert resolve_device("cpu") == "cpu" and resolve_device("mps") == "mps"
    assert resolve_device("auto") in ("mps", "cuda", "cpu")


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"通过：{test.__name__}")
    print(f"共 {len(tests)} 项测试通过")
