"""ragbase doctor：一屏环境自检，每项给出 ✓（正常）/ !（警告，不影响启动）/ ✗（必须修复）和修复提示。

- 默认不调用大模型、不修改任何文件、不加载 embedding / reranker 模型；--check-llm 才会发一次最小的大模型请求。
- 只要有一项 ✗ 就以非零码退出；警告不影响退出码。
- 绝不打印任何密钥（大模型 / API / Qdrant 的 Key 只报“是否已设置”）。
"""
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ragbase.config import Settings, project_home

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


def check_device(settings: Settings) -> list[Check]:
    try:
        from ragbase.config import resolve_device

        device = resolve_device(settings.device)
    except Exception as exc:  # torch 没装好 / 导入失败
        return [Check(FAIL, f"无法探测计算设备：{type(exc).__name__}: {exc}", "重新安装依赖：uv sync --locked")]
    source = f"RAG_DEVICE={settings.device}" if settings.device != "auto" else "自动选择"
    if device == "cpu":
        return [Check(WARN, f"计算设备：cpu（{source}）", "纯 CPU 可以运行，但精排（rerank）较慢，每个问题可能多等数秒；有 Apple Silicon（MPS）或 NVIDIA GPU（CUDA）会快很多")]
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
        return results

    results = [Check(OK, f"向量库：{backend}（{store.location}）{reach}")]
    if docs and docs[0]:
        results.append(Check(WARN, f"索引为空（文档目录里有 {docs[0]} 个文档待入库），还不能问答", "运行：ragbase ingest"))
    else:
        results.append(Check(WARN, f"索引为空，且文档目录里还没有文档（支持 PDF / docx / md / txt）：{settings.data_dir}", "放入资料后运行：ragbase ingest"))
    if docs and docs[1]:
        results.append(Check(WARN, f"有 {docs[1]} 个旧版 .doc 文件会被跳过", "请另存为 .docx"))
    return results


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

def run_checks(settings: Settings, check_llm_call: bool = False) -> list[tuple[str, list[Check]]]:
    """按分组运行全部检查；某项检查自己抛异常时记为 ✗，不影响其余项。"""
    groups: list[tuple[str, Callable[[], list[Check]]]] = [
        ("运行环境", lambda: check_python() + check_paths(settings) + check_env_file()),
        ("本地模型", lambda: check_device(settings) + check_models(settings)),
        ("知识库", lambda: check_knowledge(settings)),
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


def run_doctor(settings: Settings, check_llm_call: bool = False, out=print) -> int:
    color = sys.stdout.isatty() and "NO_COLOR" not in os.environ
    return render(run_checks(settings, check_llm_call), color=color, out=out)
