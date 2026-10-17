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


def _without_rag_env():
    """默认值测试要和外部环境无关：CI 里会设 RAG_DEVICE=cpu 等变量，先临时清掉所有 RAG_* 变量。"""
    import os
    from contextlib import contextmanager

    @contextmanager
    def scrub():
        saved = {key: value for key, value in os.environ.items() if key.startswith("RAG_")}
        for key in saved:
            del os.environ[key]
        try:
            yield
        finally:
            os.environ.update(saved)

    return scrub()


def test_defaults_match_current_behavior():
    with _without_rag_env():
        settings = _settings(llm_model=None, llm_api_key=None, llm_base_url=None)
    assert settings.device == "auto" and settings.mode == "hybrid"
    assert (settings.retrieve_k, settings.per_route_k, settings.rerank_k) == (20, 30, 5)
    assert settings.min_score == RERANK_MIN_SCORE == 0.7
    assert settings.context_min_score == CONTEXT_MIN_SCORE == 0.95
    assert (settings.host, settings.port, settings.log_level) == ("127.0.0.1", 8000, "INFO")
    assert not settings.llm_configured
    assert settings.llm_protocol == "openai" and settings.llm_temperature == 0 and settings.llm_max_tokens is None
    assert (settings.llm_timeout, settings.llm_max_retries) == (60, 2)
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
    assert settings.llm_model == "ep-test" and settings.llm_base_url == "http://llm.test/v3"
    assert settings.llm_api_key.get_secret_value() == "sk-unit-test"
    assert "sk-unit-test" not in repr(settings)
    assert settings.llm_configured


def _with_env(env: dict):
    """临时设置环境变量后构造 Settings（不读 .env）。"""
    import os
    saved = {key: os.environ.get(key) for key in env}
    os.environ.update(env)
    try:
        return _settings()
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_rag_llm_env_wins_over_ark_and_secret_hidden():
    settings = _with_env({
        "RAG_LLM_MODEL": "new-model", "RAG_LLM_API_KEY": "sk-new-secret", "RAG_LLM_BASE_URL": "http://new.test/v1",
        "ARK_CHAT_MODEL": "old-model", "ARK_CHAT_API_KEY": "sk-old-secret", "ARK_BASE_URL": "http://old.test/v3",
        "RAG_LLM_TIMEOUT": "12.5", "RAG_LLM_MAX_RETRIES": "0", "RAG_LLM_MAX_TOKENS": "256", "RAG_LLM_TEMPERATURE": "0.3",
    })
    assert settings.llm_model == "new-model" and settings.llm_base_url == "http://new.test/v1"
    assert settings.llm_api_key.get_secret_value() == "sk-new-secret"
    assert (settings.llm_timeout, settings.llm_max_retries, settings.llm_max_tokens) == (12.5, 0, 256)
    assert settings.llm_temperature == 0.3 and settings.llm_configured
    for text in (repr(settings), str(settings.model_dump()), settings.model_dump_json(), repr(settings.llm_config())):
        assert "sk-new-secret" not in text and "sk-old-secret" not in text, text


def test_llm_blank_values_mean_unset():
    settings = _settings(llm_model="  ", llm_api_key="", llm_base_url=" ")
    assert settings.llm_model is None and settings.llm_api_key is None and settings.llm_base_url is None
    assert not settings.llm_configured
    partial = _settings(llm_model="m", llm_base_url="http://x", llm_api_key=None)  # 缺密钥（显式 None：环境里可能有 .env 灌入的值）
    assert not partial.llm_configured


def test_eval_llm_overrides_fall_back_to_main():
    base = dict(llm_model="main", llm_base_url="http://main/v1", llm_api_key="sk-main")
    settings = _settings(**base)
    eval_config = settings.llm_config("eval")
    assert (eval_config.model, eval_config.base_url) == ("main", "http://main/v1")
    assert eval_config.api_key.get_secret_value() == "sk-main"
    settings = _settings(**base, eval_llm_model="judge", eval_llm_api_key="sk-judge")
    answer_config, eval_config = settings.llm_config("answer"), settings.llm_config("eval")
    assert answer_config.model == "main" and answer_config.api_key.get_secret_value() == "sk-main"
    assert eval_config.model == "judge" and eval_config.api_key.get_secret_value() == "sk-judge"
    assert eval_config.base_url == "http://main/v1"  # 没单独设置的项沿用回答模型
    assert "sk-judge" not in repr(settings)
    # 只配了评测覆盖项不算回答模型已配置
    only_eval = _settings(
        llm_model=None, llm_api_key=None, llm_base_url=None,  # rag_common 导入时会把 .env 灌进环境变量，这里显式清空
        eval_llm_model="j", eval_llm_base_url="http://j", eval_llm_api_key="k",
    )
    assert not only_eval.llm_configured and only_eval.llm_config("eval").configured


def test_invalid_values_rejected():
    for kwargs in (
        {"mode": "nope"}, {"device": "tpu"}, {"min_score": 2}, {"port": 0}, {"log_level": "loud"}, {"max_upload_mb": 0},
        {"llm_protocol": "carrier-pigeon"}, {"llm_timeout": 0}, {"llm_max_retries": -1}, {"llm_max_tokens": 0},
    ):
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
