"""开箱即用相关命令的单元测试：init / models download / doctor / web_dist 回退 / 模型名配置。

全部使用临时目录和假对象：不联网、不下载模型、不加载 embedding / reranker、不调用真实大模型、不碰真实的 data/ 与索引。

运行：uv run python tests/test_cli_setup.py
"""
import contextlib
import io
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

from ragbase import cli, config, doctor, scaffold
from ragbase.config import DEPRECATED_SETTINGS, Settings, get_settings
from ragbase.models import download

ROOT = Path(__file__).resolve().parent.parent
_SCRATCH = tempfile.TemporaryDirectory(prefix="ragdemo-setup-test-")
SCRATCH = Path(_SCRATCH.name).resolve()  # resolve：macOS 的 /var 是 /private/var 的软链，project_home() 会解析
_counter = iter(range(10_000))


def _fresh_dir(name: str) -> Path:
    path = SCRATCH / f"{name}-{next(_counter)}"
    path.mkdir()
    return path


@contextlib.contextmanager
def env(**values):
    """临时设置 / 删除环境变量（None = 删除），并先清掉所有 RAG_* / ARK_* 变量，保证与外部环境无关。"""
    saved = dict(os.environ)
    for key in list(os.environ):
        if key.startswith(("RAG_", "ARK_")):
            del os.environ[key]
    for key, value in values.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    get_settings.cache_clear()
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)
        get_settings.cache_clear()


@contextlib.contextmanager
def patched(obj, name, value):
    original = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, original)


def run_cli(*argv) -> tuple[int, str]:
    """跑一条 ragbase 子命令（不经过 load_env），返回 (退出码, 标准输出)。"""
    buffer = io.StringIO()
    code = 0
    with contextlib.redirect_stdout(buffer):
        try:
            args = cli.build_parser().parse_args(list(argv))
            args.func(args)
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 1
    return code, buffer.getvalue()


def env_var_names() -> dict[str, str]:
    """Settings 每个字段对应的环境变量名（带别名的取第一个 RAG_*），从模型本身推导，新增字段后测试自动覆盖。"""
    names = {}
    for field_name, field in Settings.model_fields.items():
        if field_name in DEPRECATED_SETTINGS:
            continue  # 已废弃的配置项不再出现在 env.example 里（文档里单独列在“已废弃”一节）
        alias = field.validation_alias
        if alias is not None and hasattr(alias, "choices"):
            choices = [c for c in alias.choices if isinstance(c, str) and c.startswith("RAG_")]
            names[field_name] = choices[0]
        else:
            names[field_name] = f"RAG_{field_name.upper()}"
    return names


# ---------------------------------------------------------------- init

def test_env_example_covers_every_setting():
    text = scaffold.read_resource("env.example")
    missing = [
        env_name for env_name in env_var_names().values()
        if not re.search(rf"^#?\s*{env_name}=", text, re.MULTILINE)
    ]
    assert not missing, f".env.example 缺少这些配置项：{missing}"
    # 模板里除 CHROMA_TELEMETRY_DISABLED 外的赋值都必须是注释：生成的 .env 不能凭空让大模型 / 鉴权变成“已配置”
    active = [line for line in text.splitlines() if re.match(r"^[A-Za-z_]+=", line)]
    assert active == ["CHROMA_TELEMETRY_DISABLED=true"], active
    assert "hf-mirror.com" in text and "RAG_API_KEY" in text and "RAG_LLM_BASE_URL" in text
    # 仓库根目录的 .env.example 与随包模板是同一份（符号链接 / 同内容）
    root_copy = ROOT / ".env.example"
    assert root_copy.is_file() and root_copy.read_text(encoding="utf-8") == text
    # 里面没有看起来像真密钥的东西
    assert not re.search(r"sk-[A-Za-z0-9]{20,}", text)


def test_init_creates_files_and_never_overwrites():
    home = _fresh_dir("init")
    actions = scaffold.init_home(home)
    assert [a.status for a in actions] == ["created"] * 4
    assert (home / "data").is_dir() and (home / "data" / "README.txt").read_text(encoding="utf-8").strip()
    assert (home / ".env").read_text(encoding="utf-8") == scaffold.read_resource("env.example")
    assert (home / ".env").stat().st_mode & 0o777 == 0o600
    assert (home / "bm25_userdict.txt").read_text(encoding="utf-8") == scaffold.read_resource("bm25_userdict.txt")
    assert "挣值分析" in (home / "bm25_userdict.txt").read_text(encoding="utf-8")

    # 再次运行：全部跳过，已有内容（包括用户改过的 .env）原样保留
    (home / ".env").write_text("RAG_LLM_MODEL=mine\n", encoding="utf-8")
    (home / "bm25_userdict.txt").write_text("我的词 100 n\n", encoding="utf-8")
    actions = scaffold.init_home(home)
    assert all(a.status == "skipped" for a in actions)
    assert (home / ".env").read_text(encoding="utf-8") == "RAG_LLM_MODEL=mine\n"
    assert (home / "bm25_userdict.txt").read_text(encoding="utf-8") == "我的词 100 n\n"
    # 带 api_key 也不会改动已有 .env
    scaffold.init_home(home, api_key="should-not-appear")
    assert "should-not-appear" not in (home / ".env").read_text(encoding="utf-8")

    # --force：覆盖，并把旧文件备份成 .bak
    actions = scaffold.init_home(home, force=True)
    assert {a.path.name: a.status for a in actions}[".env"] == "overwritten"
    assert (home / ".env.bak").read_text(encoding="utf-8") == "RAG_LLM_MODEL=mine\n"
    assert (home / "bm25_userdict.txt.bak").read_text(encoding="utf-8") == "我的词 100 n\n"
    assert (home / ".env").read_text(encoding="utf-8") == scaffold.read_resource("env.example")


def test_init_cli_api_key_and_next_steps():
    with env():
        home = _fresh_dir("init-key")
        code, out = run_cli("init", str(home), "--api-key")
        assert code == 0
        text = (home / ".env").read_text(encoding="utf-8")
        keys = re.findall(r"^RAG_API_KEY=(\S+)$", text, re.MULTILINE)
        assert len(keys) == 1 and len(keys[0]) >= 32, "应恰好写入一行 RAG_API_KEY，且足够长"
        assert keys[0] not in out, "密钥不应打印到终端"
        assert "ragbase models download" in out and "ragbase ingest" in out and "ragbase doctor" in out
        assert str(home) in out  # 目录不是当前目录时提示 cd
        # 生成的 .env 能被 Settings 正确读取：鉴权开启，大模型仍未配置
        settings = Settings(_env_file=home / ".env")
        assert settings.api_key.get_secret_value() == keys[0] and not settings.llm_configured
        # 两次生成的密钥不同
        other = _fresh_dir("init-key2")
        run_cli("init", str(other), "--api-key")
        assert re.search(r"^RAG_API_KEY=(\S+)$", (other / ".env").read_text(encoding="utf-8"), re.MULTILINE).group(1) != keys[0]
        # 已有 .env 时 --api-key 不覆盖，并明确告知
        code, out = run_cli("init", str(home), "--api-key")
        assert code == 0 and "没有写入新密钥" in out
        assert re.findall(r"^RAG_API_KEY=(\S+)$", (home / ".env").read_text(encoding="utf-8"), re.MULTILINE) == keys
        # 不带 --api-key：模板里的 RAG_API_KEY 仍是注释，鉴权关闭
        plain = _fresh_dir("init-plain")
        run_cli("init", str(plain))
        assert Settings(_env_file=plain / ".env").api_key is None


def test_init_defaults_to_rag_home():
    home = _fresh_dir("init-home")
    with env(RAG_HOME=str(home)):
        code, _ = run_cli("init")
    assert code == 0 and (home / ".env").is_file() and (home / "data" / "README.txt").is_file()


# ---------------------------------------------------------------- models download

class FakeHub:
    """假的 huggingface_hub.snapshot_download：缓存内容记在 cache 里，download 次数记在 calls 里。"""

    def __init__(self, cached=(), incomplete=()):
        self.root = _fresh_dir("hub")
        self.calls: list[str] = []
        self.fail: set[str] = set()
        self.cache: dict[str, Path] = {}
        for repo in cached:
            self.add(repo)
        for repo in incomplete:
            self.add(repo, weights=False)

    def add(self, repo: str, weights: bool = True) -> None:
        snapshot = self.root / repo.replace("/", "--")
        snapshot.mkdir(parents=True, exist_ok=True)
        (snapshot / "config.json").write_text("{}", encoding="utf-8")
        if weights:
            (snapshot / "model.safetensors").write_bytes(b"x")
        self.cache[repo] = snapshot

    def snapshot_download(self, repo_id, local_files_only=False, **_):
        if local_files_only:
            if repo_id not in self.cache:
                raise FileNotFoundError("Cannot find an appropriate cached snapshot folder")
            return str(self.cache[repo_id])
        self.calls.append(repo_id)
        if repo_id in self.fail:
            raise ConnectionError("network down")
        self.add(repo_id)
        return str(self.cache[repo_id])


def _settings(**kwargs) -> Settings:
    return Settings(_env_file=None, **kwargs)


def test_model_names_configurable():
    with env():
        settings = _settings()
        assert settings.embedding_model == "Qwen/Qwen3-Embedding-0.6B" == config.DEFAULT_EMBEDDING_MODEL
        assert settings.reranker_model == "Qwen/Qwen3-Reranker-0.6B" == config.DEFAULT_RERANKER_MODEL
    with env(RAG_EMBEDDING_MODEL=" my/embed ", RAG_RERANKER_MODEL="my/rerank"):
        settings = _settings()
        assert (settings.embedding_model, settings.reranker_model) == ("my/embed", "my/rerank")
        assert [(s.role, s.repo_id) for s in download.required_models(settings)] == [("embedding", "my/embed"), ("reranker", "my/rerank")]
    with env():
        try:
            _settings(embedding_model="  ")
        except ValueError:
            pass
        else:
            raise AssertionError("空模型名应被拒绝")


def test_models_use_configured_names():
    """Qwen3Embeddings / Qwen3Reranker 不传名字时取配置里的仓库名（用假的 SentenceTransformer / CrossEncoder）。"""
    from ragbase.models import embeddings, reranker

    seen = {}

    class FakeModel:
        def __init__(self, name, **kwargs):
            seen.setdefault("names", []).append(name)

    with env(RAG_EMBEDDING_MODEL="my/embed", RAG_RERANKER_MODEL="my/rerank", RAG_DEVICE="cpu"):
        with patched(embeddings, "SentenceTransformer", FakeModel), patched(reranker, "CrossEncoder", FakeModel):
            embeddings.Qwen3Embeddings()
            reranker.Qwen3Reranker()
            embeddings.Qwen3Embeddings(model_name="explicit/one")  # 显式传入优先
    assert seen["names"] == ["my/embed", "my/rerank", "explicit/one"]
    with env(RAG_DEVICE="cpu"):
        seen.clear()
        with patched(embeddings, "SentenceTransformer", FakeModel), patched(reranker, "CrossEncoder", FakeModel):
            embeddings.Qwen3Embeddings()
            reranker.Qwen3Reranker()
    assert seen["names"] == ["Qwen/Qwen3-Embedding-0.6B", "Qwen/Qwen3-Reranker-0.6B"]  # 默认行为不变


def test_models_download_skips_cached():
    import huggingface_hub

    with env():
        settings = _settings()
        hub = FakeHub(cached=[settings.embedding_model, settings.reranker_model])
        lines: list[str] = []
        with patched(huggingface_hub, "snapshot_download", hub.snapshot_download):
            results = download.ensure_models(settings, download=download.download_model, out=lines.append)
        assert [status for _, status in results] == ["cached", "cached"]
        assert hub.calls == [], "已缓存时不应触发任何下载"
        assert sum("已在本地缓存" in line for line in lines) == 2

        # 只缺 reranker：只下载它
        hub = FakeHub(cached=[settings.embedding_model])
        with patched(huggingface_hub, "snapshot_download", hub.snapshot_download):
            results = download.ensure_models(settings, out=lambda _l: None)
            assert hub.calls == [settings.reranker_model]
            assert [status for _, status in results] == ["cached", "downloaded"]
            assert download.missing_models(settings) == []

        # 缓存不完整（有 config 没权重）视为未缓存，需要下载
        hub = FakeHub(cached=[settings.embedding_model], incomplete=[settings.reranker_model])
        with patched(huggingface_hub, "snapshot_download", hub.snapshot_download):
            assert [s.repo_id for s in download.missing_models(settings)] == [settings.reranker_model]
            download.ensure_models(settings, out=lambda _l: None)
            assert hub.calls == [settings.reranker_model]


def test_models_download_failure_and_cli():
    import huggingface_hub

    with env():
        settings = _settings()
        hub = FakeHub()
        hub.fail.add(settings.embedding_model)
        with patched(huggingface_hub, "snapshot_download", hub.snapshot_download):
            results = download.ensure_models(settings, out=lambda _l: None)
            assert [status for _, status in results] == ["failed:ConnectionError", "downloaded"], "一个失败不应阻止另一个"
            # CLI：失败 -> 退出码 1 并提示镜像
            hub2 = FakeHub()
            hub2.fail.add(settings.reranker_model)
            with patched(huggingface_hub, "snapshot_download", hub2.snapshot_download):
                code, out = run_cli("models", "download")
            assert code == 1 and "HF_ENDPOINT=https://hf-mirror.com" in out and settings.reranker_model in out
            # CLI：全部成功 -> 0，再跑一次全部跳过
            hub3 = FakeHub()
            with patched(huggingface_hub, "snapshot_download", hub3.snapshot_download):
                code, out = run_cli("models", "download")
                assert code == 0 and "2 个新下载" in out
                hub3.calls.clear()
                code, out = run_cli("models", "download")
            assert code == 0 and hub3.calls == [] and "2 个已缓存" in out


def test_hf_home_and_endpoint_respected():
    """缓存位置 / 镜像端点由 huggingface_hub 读 HF_HOME / HF_ENDPOINT 决定（models download 直接沿用）。"""
    home = _fresh_dir("hf")
    code = "from ragbase.models.download import hub_info; print('|'.join(hub_info()))"
    result = subprocess.run(
        [sys.executable, "-I", "-c", code],
        env={**{k: v for k, v in os.environ.items() if k not in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE")},
             "HF_HOME": str(home), "HF_ENDPOINT": "https://hf-mirror.com"},
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    endpoint, cache = result.stdout.strip().split("|")
    assert endpoint == "https://hf-mirror.com" and Path(cache) == home / "hub", result.stdout


def test_serve_checks_models():
    import ragbase.ingest.common as common

    with env():
        settings = _settings()
        missing = [SimpleNamespace(role="embedding", repo_id="a/b"), SimpleNamespace(role="reranker", repo_id="c/d")]

        def serve_check(download_flag, count, miss=missing):
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                try:
                    with patched(download, "missing_models", lambda _s: miss), patched(common, "collection_count", lambda _s: count):
                        cli._ensure_models_for_serve(settings, download_flag)
                except SystemExit as exc:
                    return exc.code, buffer.getvalue()
            return None, buffer.getvalue()

        code, out = serve_check(False, 100)  # 模型缺失 + 索引非空：拦下并提示，而不是 traceback
        assert code == 1 and "ragbase models download" in out and "a/b" in out and "--download-models" in out
        assert serve_check(False, None)[0] is None and serve_check(False, 0)[0] is None  # 索引为空时不需要模型，放行
        assert serve_check(False, 100, miss=[])[0] is None  # 模型齐全：放行

        calls = []
        with patched(download, "ensure_models", lambda s, **kw: calls.append(s) or [(None, "downloaded")]):
            assert serve_check(True, 0)[0] is None and calls == [settings]  # --download-models：先下载再启动
        with patched(download, "ensure_models", lambda s, **kw: [(None, "failed:ConnectionError")]):
            code, out = serve_check(True, 0)
            assert code == 1 and "下载失败" in out
    assert cli.build_parser().parse_args(["serve", "--download-models"]).download_models is True
    assert cli.build_parser().parse_args(["serve"]).download_models is False


# ---------------------------------------------------------------- doctor

class FakeStore:
    name = "chroma"

    def __init__(self, count=42, error=None):
        self._count, self._error = count, error
        self.location = "/fake/chroma_db"
        self.closed = False

    def exists(self):
        if self._error:
            raise self._error
        return self._count > 0

    def count(self):
        return self._count

    def close(self):
        self.closed = True


def _healthy_home(name="doctor") -> Path:
    home = _fresh_dir(name)
    scaffold.init_home(home)
    (home / ".ingest_meta.json").write_text("{}", encoding="utf-8")
    (home / "web" / "dist").mkdir(parents=True)
    (home / "web" / "dist" / "index.html").write_text("<html></html>", encoding="utf-8")
    return home


def run_doctor(settings, device="mps", cached=True, store=None, **kwargs) -> tuple[int, str]:
    import ragbase.vectorstores as vectorstores

    lines: list[str] = []
    fake_snapshot = lambda repo: Path("/fake") if cached else None  # noqa: E731
    with patched(config, "resolve_device", lambda _d="auto": device), patched(download, "cached_snapshot", fake_snapshot):
        if store is not None:
            with patched(vectorstores, "create_vector_store", lambda _s=None, backend=None: store):
                code = doctor.run_doctor(settings, out=lines.append, **kwargs)
        else:
            code = doctor.run_doctor(settings, out=lines.append, **kwargs)
    return code, "\n".join(lines)


def test_doctor_healthy_prints_no_secrets():
    home = _healthy_home()
    with env(
        RAG_HOME=str(home), RAG_LLM_BASE_URL="https://llm.example.com/v1", RAG_LLM_MODEL="demo-model",
        RAG_LLM_API_KEY="sk-super-secret-123", RAG_API_KEY="auth-secret-456", RAG_QDRANT_API_KEY="qd-secret-789",
    ):
        settings = Settings(_env_file=None)
        store = FakeStore(count=42)
        code, out = run_doctor(settings, store=store)
    assert code == 0, out
    assert "✗" not in out and "共 42 个片段" in out and "demo-model" in out and "mps" in out
    assert "接口鉴权已启用" in out and "Web 界面资源就绪" in out and store.closed
    for secret in ("sk-super-secret-123", "auth-secret-456", "qd-secret-789"):
        assert secret not in out, "doctor 不应打印任何密钥"


def test_doctor_failures_set_exit_code_and_hints():
    home = _healthy_home("doctor-fail")
    with env(RAG_HOME=str(home)):
        settings = Settings(_env_file=None)
        code, out = run_doctor(settings, cached=False, store=FakeStore(count=5))
        assert code == 1 and "✗" in out
        assert "模型未缓存" in out and "ragbase models download" in out and "hf-mirror.com" in out
        assert "需要修复" in out

        code, out = run_doctor(settings, store=FakeStore(error=RuntimeError("boom")))
        assert code == 1 and "无法访问" in out and "boom" in out

        # 项目主目录不存在
        with env(RAG_HOME=str(SCRATCH / "no-such-home")):
            code, out = run_doctor(Settings(_env_file=None), store=FakeStore(count=0))
        assert code == 1 and "项目主目录不存在" in out and "ragbase init" in out


def test_doctor_warnings_do_not_fail():
    home = _fresh_dir("doctor-warn")  # 空目录：没有 .env / data / 前端 / 入库记录
    with env(RAG_HOME=str(home)):
        settings = Settings(_env_file=None)
        code, out = run_doctor(settings, device="cpu", store=FakeStore(count=0))
    assert code == 0, out
    assert "没有 .env" in out and "ragbase init" in out
    assert "cpu" in out and "较慢" in out                       # CPU 设备警告
    assert "索引为空" in out and "ragbase ingest" in out         # 空索引提示
    assert "大模型未配置" in out and "RAG_LLM_API_KEY" in out
    assert "没有 Web 界面资源" in out
    assert "✓" in out and "!" in out


def test_doctor_empty_index_with_real_chroma_path():
    """真实的 Chroma 后端 + 不存在的目录：只读探测，不创建 chroma_db，索引按空处理。"""
    home = _healthy_home("doctor-chroma")
    (home / "data" / "a.md").write_text("# 标题\n正文", encoding="utf-8")
    with env(RAG_HOME=str(home)):
        code, out = run_doctor(Settings(_env_file=None))
    assert code == 0, out
    assert "索引为空（文档目录里有 1 个文档待入库）" in out and "ragbase ingest" in out
    assert not (home / "chroma_db").exists(), "doctor 不应创建任何数据目录"


def test_doctor_auth_warning_when_exposed():
    home = _healthy_home("doctor-auth")
    with env(RAG_HOME=str(home), RAG_HOST="0.0.0.0"):
        code, out = run_doctor(Settings(_env_file=None), store=FakeStore())
    assert code == 0 and "对外开放" in out and "RAG_API_KEY" in out


def test_doctor_llm_check_only_when_requested():
    import ragbase.llm.factory as factory

    def boom(*_a, **_k):
        raise AssertionError("默认不应创建 / 调用大模型")

    home = _healthy_home("doctor-llm")
    values = dict(RAG_HOME=str(home), RAG_LLM_BASE_URL="https://llm.example.com/v1", RAG_LLM_MODEL="m", RAG_LLM_API_KEY="sk-leaky-key-000")
    with env(**values):
        settings = Settings(_env_file=None)
        with patched(factory, "create_chat_model", boom):
            code, out = run_doctor(settings, store=FakeStore())  # 默认不调用大模型
        assert code == 0 and "大模型已配置" in out and "协议 openai" in out

        calls = []
        ok = doctor.check_llm(settings, call=True, invoke=lambda: calls.append(1))
        assert calls == [1] and ok[-1].level == doctor.OK

        def failing():
            raise RuntimeError("401 Unauthorized for key sk-leaky-key-000")

        bad = doctor.check_llm(settings, call=True, invoke=failing)[-1]
        assert bad.level == doctor.FAIL and "sk-leaky-key-000" not in bad.text and "***" in bad.text
        # 经 CLI 参数传到 doctor（用假的 create_chat_model，不发请求）
        fake_model = SimpleNamespace(bind=lambda **_k: SimpleNamespace(invoke=lambda _p: "好"))
        with patched(factory, "create_chat_model", lambda _s, purpose="answer": fake_model):
            check = doctor._probe_llm(settings, "sk-leaky-key-000", None)
        assert check.level == doctor.OK
        assert cli.build_parser().parse_args(["doctor", "--check-llm"]).check_llm is True
        assert cli.build_parser().parse_args(["doctor"]).check_llm is False


def test_doctor_cli_exit_code():
    home = _healthy_home("doctor-cli")
    import ragbase.vectorstores as vectorstores

    with env(RAG_HOME=str(home)):
        with patched(config, "resolve_device", lambda _d="auto": "cpu"), patched(download, "cached_snapshot", lambda _r: None):
            with patched(vectorstores, "create_vector_store", lambda _s=None, backend=None: FakeStore()):
                code, out = run_cli("doctor")
        assert code == 1 and "模型未缓存" in out
        with patched(config, "resolve_device", lambda _d="auto": "cpu"), patched(download, "cached_snapshot", lambda _r: Path("/fake")):
            with patched(vectorstores, "create_vector_store", lambda _s=None, backend=None: FakeStore()):
                code, out = run_cli("doctor")
        assert code == 0, out


# ---------------------------------------------------------------- web_dist 回退

def test_web_dist_fallback():
    home = _fresh_dir("web-home")
    packaged = _fresh_dir("web-packaged")
    (packaged / "index.html").write_text("packaged", encoding="utf-8")
    assert config.packaged_web_dist().parts[-3:] == ("ragbase", "web", "dist")  # 真实的随包位置
    with env(RAG_HOME=str(home)), patched(config, "packaged_web_dist", lambda: packaged):
        # 主目录没有前端 -> 用随包副本
        assert _settings().web_dist == packaged
        # 主目录有前端 -> 优先主目录
        (home / "web" / "dist").mkdir(parents=True)
        assert _settings().web_dist == packaged  # 目录在但没有 index.html，不算
        (home / "web" / "dist" / "index.html").write_text("home", encoding="utf-8")
        assert _settings().web_dist == home / "web" / "dist"
        # 显式配置原样使用（相对路径按主目录解析）
        assert _settings(web_dist=packaged).web_dist == packaged
        assert _settings(web_dist=Path("custom")).web_dist == home / "custom"
    with env(RAG_HOME=str(home), RAG_WEB_DIST=str(SCRATCH / "explicit")), patched(config, "packaged_web_dist", lambda: packaged):
        assert _settings().web_dist == SCRATCH / "explicit"
    # 两处都没有 -> 仍指向主目录（服务据此提示“前端未构建”）
    empty_home, no_packaged = _fresh_dir("web-empty"), SCRATCH / "nope"
    with env(RAG_HOME=str(empty_home)), patched(config, "packaged_web_dist", lambda: no_packaged):
        assert _settings().web_dist == empty_home / "web" / "dist"


def test_web_dist_fallback_is_served():
    """回退到随包副本时，API 真能把它托管出来。"""
    from fastapi.testclient import TestClient

    from ragbase.api import app as api

    packaged = _fresh_dir("web-served")
    (packaged / "assets").mkdir()
    (packaged / "index.html").write_text("<!doctype html><title>packaged-ui</title>", encoding="utf-8")
    home = _fresh_dir("web-served-home")
    with env(RAG_HOME=str(home)), patched(config, "packaged_web_dist", lambda: packaged):
        settings = _settings()
        original = api.collection_count
        api.collection_count = lambda *a, **k: 0
        try:
            with TestClient(api.create_app(settings=settings)) as client:
                assert "packaged-ui" in client.get("/").text
        finally:
            api.collection_count = original


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"通过：{test.__name__}")
    print(f"共 {len(tests)} 项测试通过")
