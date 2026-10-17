"""本地模型（embedding + reranker）的缓存检查与下载；提供方是云端 API（或不精排）的那一个不需要下载，会被跳过。

只依赖 huggingface_hub（不导入 torch / sentence_transformers），所以 `ragbase models download`、`ragbase doctor`
和 serve 启动前的检查都很快。缓存位置跟随 HF_HOME（默认 ~/.cache/huggingface），镜像用 HF_ENDPOINT
（如 https://hf-mirror.com）；这两个变量都由 huggingface_hub 自己读取，.env 里设置也有效（命令行入口会先载入 .env）。
"""
from dataclasses import dataclass
from pathlib import Path

from ragbase.config import Settings

WEIGHT_PATTERNS = ("*.safetensors", "pytorch_model*.bin")  # 至少要有一种权重文件，才算缓存完整


@dataclass(frozen=True)
class ModelSpec:
    role: str  # 展示用：embedding / reranker
    repo_id: str
    env_name: str  # 对应的环境变量，提示用户怎么换


def required_models(settings: Settings) -> list[ModelSpec]:
    """需要在本地缓存里的模型：只包含提供方为 local 的那些。"""
    specs = []
    if settings.uses_local_embedding:
        specs.append(ModelSpec("embedding", settings.embedding_model, "RAG_EMBEDDING_MODEL"))
    if settings.uses_local_reranker:
        specs.append(ModelSpec("reranker", settings.reranker_model, "RAG_RERANKER_MODEL"))
    return specs


def cloud_roles(settings: Settings) -> list[tuple[str, str]]:
    """不需要下载的角色：[(角色, 说明)]，用于提示“云端提供方，无需下载”。"""
    roles = []
    if not settings.uses_local_embedding:
        roles.append(("embedding", f"{settings.embedding_provider} / {settings.embedding_model}"))
    if settings.reranker_provider == "api":
        roles.append(("reranker", f"api / {settings.reranker_model}"))
    elif settings.reranker_provider == "none":
        roles.append(("reranker", "none（不精排）"))
    return roles


def cached_snapshot(repo_id: str) -> Path | None:
    """仓库在本地缓存里的快照目录；没缓存或缓存明显不完整（缺 config.json / 权重文件，或软链断了）返回 None。不联网。"""
    from huggingface_hub import snapshot_download

    try:
        path = Path(snapshot_download(repo_id, local_files_only=True))
    except Exception:  # LocalEntryNotFoundError 等：没有缓存
        return None
    if not (path / "config.json").is_file():  # is_file 跟随软链，断链会返回 False
        return None
    if not any(match.is_file() for pattern in WEIGHT_PATTERNS for match in path.glob(pattern)):
        return None
    return path


def is_model_cached(repo_id: str) -> bool:
    return cached_snapshot(repo_id) is not None


def missing_models(settings: Settings) -> list[ModelSpec]:
    return [spec for spec in required_models(settings) if not is_model_cached(spec.repo_id)]


def download_model(repo_id: str) -> Path:
    """下载（或补全）一个仓库到 HF 缓存，显示进度条；失败时抛出 huggingface_hub / 网络异常。"""
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id))


def hub_info() -> tuple[str, str]:
    """(下载端点, 缓存目录)，用于在下载前告诉用户从哪里下、存到哪里。"""
    from huggingface_hub import constants

    return constants.ENDPOINT, str(constants.HF_HUB_CACHE)


def ensure_models(settings: Settings, download=download_model, out=print) -> list[tuple[ModelSpec, str]]:
    """逐个检查并下载缺失的模型，返回 [(模型, 状态)]，状态为 cached / downloaded / failed:<原因>。

    已缓存的直接跳过（不联网）；某个下载失败不影响继续处理另一个。download 可注入，便于测试。
    """
    endpoint, cache_dir = hub_info()
    results: list[tuple[ModelSpec, str]] = []
    for role, detail in cloud_roles(settings):
        out(f"[跳过] {role}：{detail}：云端提供方，无需下载")
    for spec in required_models(settings):
        if is_model_cached(spec.repo_id):
            out(f"[跳过] {spec.role}：{spec.repo_id} 已在本地缓存")
            results.append((spec, "cached"))
            continue
        out(f"[下载] {spec.role}：{spec.repo_id}（来源 {endpoint}，缓存到 {cache_dir}）")
        try:
            download(spec.repo_id)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            out(f"[失败] {spec.role}：{spec.repo_id}：{type(exc).__name__}: {exc}")
            results.append((spec, f"failed:{type(exc).__name__}"))
            continue
        if is_model_cached(spec.repo_id):
            out(f"[完成] {spec.role}：{spec.repo_id}")
            results.append((spec, "downloaded"))
        else:
            out(f"[失败] {spec.role}：{spec.repo_id}：下载结束但缓存不完整（缺少 config.json 或权重文件）")
            results.append((spec, "failed:incomplete"))
    return results
