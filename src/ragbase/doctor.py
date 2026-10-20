"""ragbase doctor：一屏环境自检，每项给出 ✓（正常）/ !（警告，不影响启动）/ ✗（必须修复）和修复提示。

- 默认不调用大模型 / embedding / rerank 接口、不修改任何文件、不加载 embedding / reranker 模型；
  --check-llm 才会发一次最小的大模型请求，--check-models 才会对每个云端 embedding / rerank 接口各发一次最小请求。
- 只要有一项 ✗ 就以非零码退出；警告不影响退出码。
- 绝不打印任何密钥（大模型 / embedding / rerank / API / Qdrant 的 Key 只报“是否已设置”）。
"""
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ragbase.config import DEFAULT_RERANKER_MODEL, RERANK_MIN_SCORE, Settings, db_file, project_home

OK, WARN, FAIL = "ok", "warn", "fail"
MIN_PYTHON = (3, 12)
SYMBOLS = {OK: "✓", WARN: "!", FAIL: "✗"}
COLORS = {OK: "32", WARN: "33", FAIL: "31"}


@dataclass(frozen=True)
class Check:
    level: str
    text: str
    hint: str = ""


# ---------------------------------------------------------------- 各项检查

def check_python() -> list[Check]:
    version = ".".join(map(str, sys.version_info[:3]))
    if sys.version_info[:2] < MIN_PYTHON:
        return [Check(FAIL, f"Python {version}（需要 >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]}）", "用 uv 安装：uv python install 3.12，然后 uv sync")]
    return [Check(OK, f"Python {version}")]


def _writable_dir(path: Path) -> bool:
    return path.is_dir() and os.access(path, os.W_OK | os.X_OK)


def check_paths(settings: Settings) -> list[Check]:
    home = project_home()
    results = []
    if not home.exists():
        results.append(Check(FAIL, f"项目主目录不存在：{home}", "运行：ragbase init"))
    elif not _writable_dir(home):
        results.append(Check(FAIL, f"项目主目录不可写：{home}", "检查目录权限，或用 RAG_HOME 指向可写目录"))
    else:
        results.append(Check(OK, f"项目主目录可写：{home}（RAG_HOME {'已设置' if os.environ.get('RAG_HOME') else '未设置，使用当前目录'}）"))
    data_dir = settings.data_dir
    if not data_dir.is_dir():
        results.append(Check(WARN, f"文档目录不存在：{data_dir}", "运行：ragbase init （或手动创建该目录）"))
    elif not _writable_dir(data_dir):
        results.append(Check(FAIL, f"文档目录不可写：{data_dir}", "检查目录权限（上传文档需要写入）"))
    else:
        results.append(Check(OK, f"文档目录：{data_dir}"))
    return results


def check_env_file() -> list[Check]:
    from ragbase.config import env_file

    path = env_file()
    if path.is_file():
        return [Check(OK, f".env 已存在：{path}")]
    return [Check(WARN, f"没有 .env：{path}", "运行：ragbase init （生成模板），然后填写 LLM 配置；也可以直接用环境变量配置")]


def _endpoint_checks(settings: Settings, role: str, endpoint, url_env: str, key_env: str) -> list[Check]:
    """云端接口的地址 / 密钥是否配置（只报是否设置，不打印密钥）。"""
    if not endpoint.base_url:
        return [Check(FAIL, f"{role}：未配置接口地址", f"设置 {url_env}（或让该服务与大模型共用，设置 RAG_LLM_BASE_URL）")]
    results = []
    if not endpoint.base_url.lower().startswith(("http://", "https://")):
        return [Check(FAIL, f"{role}：接口地址 {_strip_userinfo(endpoint.base_url)} 必须以 http:// 或 https:// 开头", f"检查 {url_env}")]
    source = "（沿用 RAG_LLM_BASE_URL）" if endpoint.base_url_from_llm else ""
    if endpoint.api_key is not None:
        key_note = "沿用 RAG_LLM_API_KEY" if endpoint.api_key_from_llm else "已设置"
        results.append(Check(OK, f"{role}：地址 {_strip_userinfo(endpoint.base_url)}{source}，密钥{key_note}"))
    else:
        results.append(Check(
            WARN,
            f"{role}：地址 {_strip_userinfo(endpoint.base_url)}{source}，未设置密钥",
            f"云端服务通常需要密钥：设置 {key_env}；本机 Ollama / vLLM 等可忽略。（地址与大模型不同源时不会沿用 RAG_LLM_API_KEY）",
        ))
    return results


def check_providers(
    settings: Settings,
    call: bool = False,
    embed_probe: Callable | None = None,
    rerank_probe: Callable | None = None,
) -> list[Check]:
    """embedding / reranker 的提供方与模型、云端地址 / 密钥是否配置、RAG_MIN_SCORE 是否需要重新标定；call=True 才发最小请求。"""
    results = []
    # ---- embedding
    if settings.embedding_provider == "local":
        results.append(Check(OK, f"embedding：本地模型 {settings.embedding_model}"))
    else:
        dims = f"，dimensions={settings.embedding_dimensions}" if settings.embedding_dimensions else ""
        results.append(Check(OK, f"embedding：云端（{settings.embedding_provider}）模型 {settings.embedding_model}{dims}"))
        results += _endpoint_checks(settings, "embedding", settings.embedding_endpoint(), "RAG_EMBEDDING_BASE_URL", "RAG_EMBEDDING_API_KEY")
    # ---- reranker
    if settings.reranker_provider == "local":
        results.append(Check(OK, f"reranker：本地模型 {settings.reranker_model}"))
    elif settings.reranker_provider == "api":
        results.append(Check(OK, f"reranker：云端（api）模型 {settings.reranker_model}"))
        results += _endpoint_checks(settings, "reranker", settings.reranker_endpoint(), "RAG_RERANKER_BASE_URL", "RAG_RERANKER_API_KEY")
    else:
        results.append(Check(
            WARN,
            "reranker：未启用精排（RAG_RERANKER_PROVIDER=none）：不会按相关度拒答，只有空召回才拒答",
            "检索质量与拒答能力都会下降；有条件请用本地或云端 reranker",
        ))
    if settings.reranker_provider != "none":
        default_reranker = settings.reranker_provider == "local" and settings.reranker_model == DEFAULT_RERANKER_MODEL
        if not default_reranker and settings.min_score == RERANK_MIN_SCORE:
            results.append(Check(
                WARN,
                f"reranker 不是默认的本地 {DEFAULT_RERANKER_MODEL}，但 RAG_MIN_SCORE 仍是默认 {RERANK_MIN_SCORE}（它是按默认模型的得分分布标定的）",
                "建议用 scripts/calibrate_rerank_threshold.py 重新标定 RAG_MIN_SCORE / RAG_CONTEXT_MIN_SCORE",
            ))
    from ragbase.models.factory import provider_problems

    if call and not provider_problems(settings):
        results += _probe_providers(settings, embed_probe, rerank_probe)
    return results


def _probe_providers(settings: Settings, embed_probe: Callable | None, rerank_probe: Callable | None) -> list[Check]:
    """--check-models：对每个云端 embedding / rerank 接口各发一次最小请求；错误信息里的密钥会被抹掉。"""
    from ragbase.models.api_client import ApiClient
    from ragbase.models.api_embeddings import OpenAIEmbeddings
    from ragbase.models.api_reranker import ApiReranker

    results = []
    jobs = []
    if settings.embedding_provider == "openai":
        jobs.append(("embedding", settings.embedding_endpoint(), embed_probe, lambda client: OpenAIEmbeddings(
            client, settings.embedding_model, dimensions=settings.embedding_dimensions, batch_size=1).probe(), "RAG_EMBEDDING_*"))
    if settings.reranker_provider == "api":
        jobs.append(("reranker", settings.reranker_endpoint(), rerank_probe, lambda client: ApiReranker(client, settings.reranker_model).probe(), "RAG_RERANKER_*"))
    for role, endpoint, probe, default_probe, env_prefix in jobs:
        secret = endpoint.api_key.get_secret_value() if endpoint.api_key else ""
        try:
            if probe is not None:
                detail = probe()
            else:
                client = ApiClient(endpoint, role)
                try:
                    detail = default_probe(client)
                finally:
                    client.close()
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            message = (message.replace(secret, "***") if secret else message)[:300]
            results.append(Check(FAIL, f"{role} 接口调用失败：{message}", f"检查 {env_prefix} 的地址（通常以 /v1 结尾）、密钥和模型名"))
        else:
            extra = f"，向量维度 {detail}" if role == "embedding" and isinstance(detail, int) else ""
            results.append(Check(OK, f"{role} 接口调用成功（一次最小请求{extra}）"))
    return results


def check_device(settings: Settings) -> list[Check]:
    try:
        from ragbase.config import resolve_device

        device = resolve_device(settings.device)
    except Exception as exc:  # torch 没装好 / 导入失败
        return [Check(FAIL, f"无法探测计算设备：{type(exc).__name__}: {exc}", "重新安装依赖：uv sync --locked")]
    source = f"RAG_DEVICE={settings.device}" if settings.device != "auto" else "自动选择"
    if device == "cpu":
        return [Check(WARN, f"计算设备：cpu（{source}）", "纯 CPU 可以运行，但精排（rerank）较慢，每个问题可能多等数秒；有 Apple Silicon（MPS）或 NVIDIA GPU（CUDA）会快很多，也可改用云端 embedding / reranker（见 docs/models.md）")]
    return [Check(OK, f"计算设备：{device}（{source}）")]


def check_models(settings: Settings) -> list[Check]:
    from ragbase.models.download import hub_info, required_models, cached_snapshot

    results = []
    for spec in required_models(settings):
        snapshot = cached_snapshot(spec.repo_id)
        if snapshot is not None:
            results.append(Check(OK, f"模型已缓存：{spec.role} {spec.repo_id}"))
        else:
            endpoint, cache_dir = hub_info()
            results.append(Check(FAIL, f"模型未缓存：{spec.role} {spec.repo_id}（缓存目录 {cache_dir}）", "运行：ragbase models download （国内网络先设置 HF_ENDPOINT=https://hf-mirror.com）"))
    return results


def _count_documents(settings: Settings) -> tuple[int, int] | None:
    """(可入库文档数, 被跳过的旧版 .doc 数)；文档目录不存在或读不了返回 None。"""
    if not settings.data_dir.is_dir():
        return None
    from ragbase.ingest.parser import list_documents

    try:
        files, skipped = list_documents(settings.data_dir)
    except OSError:
        return None
    return len(files), len(skipped)


def check_knowledge(settings: Settings) -> list[Check]:
    """向量库是否可用、索引里有多少片段，以及是否需要 ragbase ingest（同一个原因只给一条提示）。"""
    from ragbase.vectorstores import create_vector_store

    backend = settings.vector_store
    try:
        store = create_vector_store(settings)
    except Exception as exc:
        return [Check(FAIL, f"向量库（{backend}）配置无效：{exc}", "检查 RAG_VECTOR_STORE / RAG_QDRANT_URL 等设置")]
    try:
        exists = store.exists()
        count = store.count() if exists else 0
    except Exception as exc:
        if backend == "qdrant" and settings.qdrant_url:
            hint = "检查 Qdrant 服务是否已启动、RAG_QDRANT_URL 是否正确"
        elif backend == "qdrant":
            hint = "若 ragbase serve / ingest 正在运行，请先停止（嵌入式 Qdrant 同一目录只允许一个进程访问）"
        else:
            hint = "检查目录权限与磁盘空间；若目录是别的版本创建的，可备份后删除重建"
        return [Check(FAIL, f"向量库（{backend}，{store.location}）无法访问：{type(exc).__name__}: {exc}", hint)]
    finally:
        try:
            store.close()
        except Exception:
            pass

    reach = "可连接" if backend == "qdrant" and settings.qdrant_url else "可访问"
    docs = _count_documents(settings)
    if count:
        results = [Check(OK, f"向量库：{backend}（{store.location}）{reach}，共 {count} 个片段")]
        if not settings.meta_file.is_file():
            results.append(Check(WARN, f"索引非空但没有入库记录 {settings.meta_file.name}", "下次运行 ragbase ingest 会把所有文档视为新文档并重新处理"))
        results += check_index_signature(settings, store)
        return results

    results = [Check(OK, f"向量库：{backend}（{store.location}）{reach}")]
    if docs and docs[0]:
        results.append(Check(WARN, f"索引为空（文档目录里有 {docs[0]} 个文档待入库），还不能问答", "运行：ragbase ingest"))
    else:
        results.append(Check(WARN, f"索引为空，且文档目录里还没有文档（支持 PDF / docx / md / txt）：{settings.data_dir}", "放入资料后运行：ragbase ingest"))
    if docs and docs[1]:
        results.append(Check(WARN, f"有 {docs[1]} 个旧版 .doc 文件会被跳过", "请另存为 .docx"))
    return results


def check_index_signature(settings: Settings, store) -> list[Check]:
    """索引的 embedding 签名（提供方 / 模型 / 维度）与当前配置是否一致（只读，不补写签名文件）。"""
    from ragbase.index_info import IndexMismatchError, check_index

    try:
        result = check_index(settings, store, write_legacy=False)
    except IndexMismatchError as exc:
        return [Check(FAIL, str(exc))]
    if result.status == "mismatch":
        return [Check(
            FAIL,
            f"当前 embedding 配置与索引不一致（索引：{result.stored.describe()}；当前：{result.current.describe()}）",
            "运行：ragbase ingest --full 重建索引（或把 embedding 配置改回索引建立时的值）；不重建 serve / ask / chat 会拒绝启动",
        )]
    if result.status == "legacy_default":
        return [Check(OK, f"索引 embedding：{result.stored.describe()}（旧索引没有签名记录，按默认本地模型处理；下次启动 / 入库时自动补记）")]
    return [Check(OK, f"索引 embedding 与当前配置一致：{result.stored.describe()}")]


def check_llm(settings: Settings, call: bool = False, invoke: Callable | None = None) -> list[Check]:
    config = settings.llm_config("answer")
    if not config.configured:
        missing = [name for name, value in (("RAG_LLM_BASE_URL", config.base_url), ("RAG_LLM_API_KEY", config.api_key), ("RAG_LLM_MODEL", config.model)) if not value]
        return [Check(WARN, f"大模型未配置（缺 {'、'.join(missing)}）：只能检索，不能问答 / 对话", "在 .env 里填写 RAG_LLM_BASE_URL / RAG_LLM_API_KEY / RAG_LLM_MODEL（任意 OpenAI 兼容服务）")]
    results = [Check(OK, f"大模型已配置：协议 {config.protocol}，模型 {config.model}，地址 {_strip_userinfo(config.base_url)}，密钥已设置")]
    if call:
        results.append(_probe_llm(settings, config.api_key.get_secret_value(), invoke))
    return results


def _strip_userinfo(url: str) -> str:
    """地址里万一带了 user:pass@，展示时去掉。"""
    from urllib.parse import urlsplit, urlunsplit

    parts = urlsplit(url)
    if parts.username or parts.password:
        host = parts.hostname or ""
        netloc = f"{host}:{parts.port}" if parts.port else host
        return urlunsplit(parts._replace(netloc=netloc))
    return url


def _probe_llm(settings: Settings, secret: str, invoke: Callable | None) -> Check:
    """--check-llm：发一次最小请求（只要求回一个字），验证地址 / 密钥 / 模型名。错误信息里的密钥会被抹掉。"""
    try:
        if invoke is None:
            from ragbase.llm.factory import create_chat_model

            model = create_chat_model(settings)
            invoke = lambda: model.bind(max_tokens=8).invoke("请只回复：好")  # noqa: E731
        invoke()
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}".replace(secret, "***")[:300]
        return Check(FAIL, f"大模型调用失败：{message}", "检查 RAG_LLM_BASE_URL（通常以 /v1 结尾）、RAG_LLM_API_KEY 和模型名是否正确")
    return Check(OK, "大模型调用成功（一次最小请求）")


def _nearest_existing_dir(path: Path) -> Path | None:
    """path 本身或最近的已存在祖先目录；一路都不是目录则返回 None。"""
    for candidate in (path, *path.parents):
        if candidate.exists():
            return candidate if candidate.is_dir() else None
    return None


def check_database(settings: Settings) -> list[Check]:
    """内置 SQLite 数据库：能打开、表结构版本是最新。只读探测——不创建文件、不迁移（建库与升级在 ragbase serve 启动时自动完成）。"""
    import sqlite3
    from urllib.parse import quote

    from ragbase.db import MIGRATIONS

    path = db_file(settings)
    latest = len(MIGRATIONS)
    if not path.exists():
        base = _nearest_existing_dir(path.parent)
        if base is None or not _writable_dir(base):
            return [Check(FAIL, f"数据库无法创建：{path}（目录 {base or path.parent} 不可写）", "检查目录权限，或用 RAG_DB_PATH 指向可写位置；ragbase serve 需要它来保存会话")]
        return [Check(OK, f"数据库尚未创建：{path}（首次运行 ragbase serve 时自动创建）")]
    if not path.is_file():
        return [Check(FAIL, f"数据库路径不是文件：{path}", "检查 RAG_DB_PATH")]
    if not (os.access(path, os.W_OK) and _writable_dir(path.parent)):
        return [Check(FAIL, f"数据库不可写：{path}（WAL 模式还需要所在目录可写）", "检查文件与目录权限；ragbase serve 需要写入它来保存会话")]
    try:
        conn = sqlite3.connect(f"file:{quote(str(path))}?mode=ro", uri=True, timeout=5)
        try:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            conn.execute("SELECT count(*) FROM sqlite_master").fetchone()  # 文件损坏时在这里报错
        finally:
            conn.close()
    except sqlite3.Error as exc:
        return [Check(FAIL, f"数据库无法打开：{path}（{exc}）", "文件可能已损坏：备份后删除它（会丢失会话与体检记录），重启后自动重建；或用 RAG_DB_PATH 换一个位置")]
    if version == latest:
        return [Check(OK, f"数据库：{path}，表结构版本 {version}（最新）")]
    if version < latest:
        return [Check(WARN, f"数据库表结构版本 {version}，当前程序需要 {latest}：{path}", "无需手动处理，下次运行 ragbase serve 会自动升级（升级前建议先备份该文件）")]
    return [Check(FAIL, f"数据库表结构版本 {version} 比当前程序支持的 {latest} 更新：{path}", "请升级 ragbase；不要用旧版本打开新版本的数据库")]


def check_auth(settings: Settings) -> list[Check]:
    from ragbase.api.auth import is_loopback_host

    if settings.api_key is not None:
        return [Check(OK, "接口鉴权已启用（RAG_API_KEY 已设置）")]
    if is_loopback_host(settings.host):
        return [Check(OK, f"接口鉴权未启用（监听 {settings.host}，仅本机可访问）")]
    return [Check(WARN, f"接口鉴权未启用，但监听地址是 {settings.host}（对外开放）", "设置 RAG_API_KEY：ragbase init --api-key 会生成一把随机密钥（或 openssl rand -hex 32）")]


def check_web(settings: Settings) -> list[Check]:
    index = settings.web_dist / "index.html"
    if index.is_file():
        return [Check(OK, f"Web 界面资源就绪：{settings.web_dist}")]
    return [Check(WARN, f"没有 Web 界面资源：{settings.web_dist}", "仓库里运行 scripts/build_web.sh（需要 Node 22 + pnpm）；API 仍可正常使用。Docker 镜像已内置界面")]


# ---------------------------------------------------------------- 汇总与输出

def _local_model_checks(settings: Settings) -> list[Check]:
    """只有用到本地模型时才检查设备和模型缓存（全云端配置不探测 torch）。"""
    if not settings.uses_local_models:
        return [Check(OK, "未使用本地模型：跳过设备与模型缓存检查")]
    return check_device(settings) + check_models(settings)


def run_checks(
    settings: Settings, check_llm_call: bool = False, check_models_call: bool = False
) -> list[tuple[str, list[Check]]]:
    """按分组运行全部检查；某项检查自己抛异常时记为 ✗，不影响其余项。"""
    groups: list[tuple[str, Callable[[], list[Check]]]] = [
        ("运行环境", lambda: check_python() + check_paths(settings) + check_env_file()),
        ("模型（embedding / reranker）", lambda: check_providers(settings, call=check_models_call) + _local_model_checks(settings)),
        ("知识库", lambda: check_knowledge(settings)),
        ("数据库（会话、题集、体检报告）", lambda: check_database(settings)),
        ("大模型", lambda: check_llm(settings, call=check_llm_call)),
        ("服务", lambda: check_auth(settings) + check_web(settings)),
    ]
    results = []
    for title, func in groups:
        try:
            checks = func()
        except Exception as exc:
            checks = [Check(FAIL, f"检查本身出错：{type(exc).__name__}: {exc}")]
        results.append((title, checks))
    return results


def render(results: list[tuple[str, list[Check]]], color: bool = False, out=print) -> int:
    """打印结果，返回退出码（有 ✗ 为 1）。"""
    counts = {OK: 0, WARN: 0, FAIL: 0}

    def paint(level: str, text: str) -> str:
        return f"\033[{COLORS[level]}m{text}\033[0m" if color else text

    out("ragbase doctor")
    for title, checks in results:
        out(f"\n[{title}]")
        for check in checks:
            counts[check.level] += 1
            out(f"  {paint(check.level, SYMBOLS[check.level])} {check.text}")
            if check.hint:
                out(f"      → {check.hint}")
    out(f"\n共 {sum(counts.values())} 项：{counts[OK]} 项正常，{counts[WARN]} 项警告，{counts[FAIL]} 项需要修复")
    if counts[FAIL]:
        out("存在 ✗ 项，请先按提示修复。")
    elif counts[WARN]:
        out("没有阻塞问题；! 项按提示处理后体验更完整。")
    else:
        out("一切就绪。")
    return 1 if counts[FAIL] else 0


def run_doctor(settings: Settings, check_llm_call: bool = False, out=print, check_models_call: bool = False) -> int:
    color = sys.stdout.isatty() and "NO_COLOR" not in os.environ
    return render(run_checks(settings, check_llm_call, check_models_call), color=color, out=out)
