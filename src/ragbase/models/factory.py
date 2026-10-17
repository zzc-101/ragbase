"""模型工厂：按 Settings 创建 embedding / reranker，是项目里唯一构造它们的地方（build_pipeline、入库、命令行、评测都走这里）。

- embedding：local（Qwen3Embeddings，sentence-transformers）| openai（OpenAI 兼容 /embeddings）
- reranker：local（Qwen3Reranker，CrossEncoder）| api（/rerank）| none（不精排）
- torch / sentence_transformers / transformers 只在本地实现里导入（这里都是函数内懒加载），
  所以全云端配置启动时不会加载它们；本模块可在任何脚本里放心导入。
- 配置缺项（云端地址没填等）抛 ProviderConfigError，消息是可直接展示的中文说明。
"""
from ragbase.config import EMBEDDING_PROVIDERS, RERANKER_PROVIDERS, ApiEndpoint, Settings, resolve_device
from ragbase.models.api_client import ApiClient


class ProviderConfigError(ValueError):
    """模型提供方配置不完整或无效。"""


def _check_endpoint(endpoint: ApiEndpoint, role: str, url_env: str) -> list[str]:
    problems = []
    if not endpoint.base_url:
        problems.append(f"缺少接口地址：请设置 {url_env}（或 RAG_LLM_BASE_URL，该服务同时提供 {role} 时可共用）")
    elif not endpoint.base_url.lower().startswith(("http://", "https://")):
        problems.append(f"{url_env} 必须以 http:// 或 https:// 开头")
    return problems


def provider_problems(settings: Settings) -> list[str]:
    """当前 embedding / reranker 配置里会导致无法创建的问题（中文，不含密钥）；没有问题返回空列表。不联网。"""
    problems = []
    if settings.embedding_provider not in EMBEDDING_PROVIDERS:  # Settings 已校验，防御性检查
        problems.append(f"不支持的 RAG_EMBEDDING_PROVIDER={settings.embedding_provider!r}，可选：{'、'.join(EMBEDDING_PROVIDERS)}")
    elif settings.embedding_provider == "openai":
        problems += [
            f"embedding：{text}"
            for text in _check_endpoint(settings.embedding_endpoint(), "embedding", "RAG_EMBEDDING_BASE_URL")
        ]
    if settings.reranker_provider not in RERANKER_PROVIDERS:
        problems.append(f"不支持的 RAG_RERANKER_PROVIDER={settings.reranker_provider!r}，可选：{'、'.join(RERANKER_PROVIDERS)}")
    elif settings.reranker_provider == "api":
        problems += [
            f"reranker：{text}"
            for text in _check_endpoint(settings.reranker_endpoint(), "rerank", "RAG_RERANKER_BASE_URL")
        ]
    return problems


def resolve_local_device(settings: Settings, device: str | None = None) -> str:
    """有本地模型时解析计算设备（会导入 torch）；全云端返回 "cloud"（不导入 torch）。"""
    if not settings.uses_local_models:
        return "cloud"
    return resolve_device(device or settings.device)


def _explicit_device(settings: Settings, device: str | None) -> str | None:
    """传给本地模型的设备：显式给定的优先，其次配置里非 auto 的；auto 交给模型类自己解析（避免这里提前导入 torch）。"""
    return device or (None if settings.device == "auto" else settings.device)


def create_embeddings(settings: Settings, device: str | None = None):
    """按配置创建 embedding（embed_documents / embed_query / embed_queries）。device 只对本地提供方有意义。"""
    problems = [text for text in provider_problems(settings) if text.startswith("embedding：")]
    if problems:
        raise ProviderConfigError("；".join(problems))
    if settings.embedding_provider == "local":
        from ragbase.models.embeddings import Qwen3Embeddings

        return Qwen3Embeddings(model_name=settings.embedding_model, device=_explicit_device(settings, device))
    from ragbase.models.api_embeddings import OpenAIEmbeddings

    return OpenAIEmbeddings(
        ApiClient(settings.embedding_endpoint(), "embedding"),
        settings.embedding_model,
        dimensions=settings.embedding_dimensions,
        batch_size=settings.embedding_batch_size,
        query_instruction=settings.embedding_query_instruction,
    )


def create_reranker(settings: Settings, device: str | None = None):
    """按配置创建 reranker（rerank / rerank_with_scores）。provider=none 返回不精排的实现（分数为 None）。"""
    problems = [text for text in provider_problems(settings) if text.startswith("reranker：")]
    if problems:
        raise ProviderConfigError("；".join(problems))
    if settings.reranker_provider == "local":
        from ragbase.models.reranker import Qwen3Reranker

        return Qwen3Reranker(model_name=settings.reranker_model, device=_explicit_device(settings, device))
    from ragbase.models.api_reranker import ApiReranker, NoReranker

    if settings.reranker_provider == "none":
        return NoReranker()
    return ApiReranker(ApiClient(settings.reranker_endpoint(), "rerank"), settings.reranker_model)
