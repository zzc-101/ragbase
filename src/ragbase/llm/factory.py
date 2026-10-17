"""大模型工厂：按 Settings 创建聊天模型，是项目里唯一构造 ChatOpenAI / AsyncOpenAI 的地方。

- create_chat_model：LangChain 聊天模型（回答链、评测里的中文裁判）。
- create_ragas_llm：ragas 评测用的 LLM 包装（与上面同一套配置，仅在评测时才 import ragas）。
- purpose="answer" 用回答模型配置；purpose="eval" 用 eval_llm_* 覆盖项（未设置的项沿用回答模型）。

新增协议（如 anthropic）：config.Settings.llm_protocol 加一个字面量，再在下面各工厂函数里加一个分支即可，
调用方都不用改。协议无法识别时在构造阶段（服务启动）就抛 ValueError，而不是等到第一次请求。
"""
from langchain_core.language_models import BaseChatModel

from ragbase.config import LLMConfig, LlmPurpose, Settings

SUPPORTED_PROTOCOLS = ("openai",)


def _check_protocol(config: LLMConfig) -> None:
    if config.protocol not in SUPPORTED_PROTOCOLS:
        raise ValueError(f"不支持的大模型协议 {config.protocol!r}（RAG_LLM_PROTOCOL），目前支持：{'、'.join(SUPPORTED_PROTOCOLS)}")


def create_chat_model(settings: Settings, purpose: LlmPurpose = "answer") -> BaseChatModel | None:
    """按配置创建聊天模型；模型名 / 地址 / 密钥没配齐返回 None（服务仍可只检索）；协议不支持抛 ValueError。

    超时与重试由底层客户端处理：llm_timeout 是单次请求超时，llm_max_retries 是超时 / 连接错误 / 429 / 5xx 的
    自动重试次数（指数退避），所以最坏等待约 llm_timeout * (llm_max_retries + 1) 再加退避间隔。
    """
    config = settings.llm_config(purpose)
    _check_protocol(config)
    if not config.configured:
        return None
    if config.protocol == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout=config.timeout,
            max_retries=config.max_retries,
        )
    raise ValueError(f"未实现的大模型协议：{config.protocol!r}")  # SUPPORTED_PROTOCOLS 与上面的分支不同步时才会到这


def create_ragas_llm(settings: Settings, cache=None):
    """创建 ragas 用的 LLM 包装；配置不全返回 None。cache 为 ragas 的缓存后端（可选）。"""
    config = settings.llm_config("eval")
    _check_protocol(config)
    if not config.configured:
        return None
    if config.protocol == "openai":
        from openai import AsyncOpenAI
        from ragas.llms import llm_factory

        client = AsyncOpenAI(
            api_key=config.api_key.get_secret_value(),
            base_url=config.base_url,
            timeout=config.timeout,
            max_retries=config.max_retries,
        )
        options = {"max_tokens": config.max_tokens} if config.max_tokens else {}
        return llm_factory(
            model=config.model,
            provider="openai",
            client=client,
            cache=cache,
            temperature=config.temperature,
            **options,
        )
    raise ValueError(f"未实现的大模型协议：{config.protocol!r}")
