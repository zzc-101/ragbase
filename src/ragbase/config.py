"""集中配置：环境变量 / .env -> Settings，命令行与 API 服务共用，保证默认值只有一个来源。

- 自有配置统一加 RAG_ 前缀（如 RAG_DEVICE、RAG_PORT）；大模型用 RAG_LLM_*，旧的 ARK_* 变量名仍然兼容（RAG_LLM_* 优先）。
- API Key 用 SecretStr 保存，repr / 日志里只会显示掩码；读取明文请用 get_secret_value()。
- 本模块不 import torch 等重型依赖（resolve_device 里才懒加载），可在任何脚本里放心导入。
"""
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from dotenv import load_dotenv
from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


def project_home() -> Path:
    """项目主目录：环境变量 RAG_HOME，未设置则为当前工作目录。

    data/、chroma_db/、qdrant_db/、.ingest_meta.json、.bm25_cache/、web/dist、bm25_userdict.txt、eval_questions.json、.env
    的默认位置都从这里推导（在仓库根目录运行时与旧版布局完全一致）。
    注意 RAG_HOME 必须是真正的环境变量：.env 的位置本身依赖它，所以不能写在 .env 里。
    每次调用都重新读取环境变量（不缓存），便于测试里临时切换。
    """
    return Path(os.environ.get("RAG_HOME") or Path.cwd()).expanduser().resolve()


def env_file() -> Path:
    return project_home() / ".env"


def userdict_file() -> Path:
    """BM25 用户词典；放在主目录而不是 data/ 下：data/*.txt 会被当文档入库。"""
    return project_home() / "bm25_userdict.txt"


def eval_questions_file() -> Path:
    return project_home() / "eval_questions.json"


def ragas_cache_dir() -> Path:
    return project_home() / ".ragas_cache"


def packaged_web_dist() -> Path:
    """随包分发的前端构建产物位置（scripts/build_web.sh 复制到这里；wheel / Docker 镜像里带着它）。"""
    return Path(__file__).resolve().parent / "web" / "dist"


def default_web_dist() -> Path:
    """前端目录的默认值：优先主目录下的 web/dist（仓库里 pnpm build 的产物），没有就退到随包的前端副本，都没有仍指向前者（便于提示）。

    只在创建 Settings 时判断一次；显式设置 RAG_WEB_DIST 则原样使用，不走这里。
    """
    home_dist = project_home() / "web" / "dist"
    if (home_dist / "index.html").is_file():
        return home_dist
    packaged = packaged_web_dist()
    return packaged if (packaged / "index.html").is_file() else home_dist


def load_env() -> None:
    """把主目录下的 .env 载入 os.environ（不覆盖已有的环境变量）；文件不存在时什么也不做。"""
    load_dotenv(env_file())


# 最相关片段的精排得分（P(相关)，0~1）低于该值就拒答、不调用 LLM。
# 校准（scripts/calibrate_rerank_threshold.py）：45 道库内问题 top1 得分 0.937~0.9999（中位 0.9994），
# 21 道库外问题 0.0000~0.9882（14 道 < 0.06，其余 0.61~0.99 与库内有重叠）。
# 取最低库内分 × 0.75 ≈ 0.7：库内零误拒，拦下 15/21；宁可放过也不误拒。
RERANK_MIN_SCORE = 0.7

# 通过拒答门槛后，只把 rerank_score >= 该值的片段交给 LLM（最相关的 top1 总是保留）；None/0 不过滤。
# 数据（临时实验脚本，30 道评测题、hybrid、精排 Top5、不调用 LLM）：Reranker 得分高度饱和，
# 库内题全部 5 个片段都 >= 0.96（Top5 最低 0.9612），所以 <= 0.95 的阈值对库内题等于不过滤：
#   阈值    平均保留片段   要点召回(point_recall)
#   0~0.95     5.00          0.8344（与不过滤相同）
#   0.97       4.87          0.8210（-0.013，超出 0.01 容忍）
#   0.99       4.63          0.7988
#   0.998      3.37          0.7121
# 取满足“要点召回不低于未过滤值 0.01”的最高候选 0.95：库内题零损失，只在片段得分明显偏低时才裁剪。
# 注意得分已饱和，提高阈值会开始裁掉对比 / 多方面题需要的片段，别盲目调高。
CONTEXT_MIN_SCORE = 0.95

# 本地模型的 Hugging Face 仓库名；RAG_EMBEDDING_MODEL / RAG_RERANKER_MODEL 可改
# （提供方为云端 API 时，同名变量是发给服务商的模型 ID）。
# 换 embedding 模型（或提供方 / 维度）会让向量不兼容，必须 ragbase ingest --full 重建索引；换 reranker 需要重新校准 RAG_MIN_SCORE。
DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_RERANKER_MODEL = "Qwen/Qwen3-Reranker-0.6B"

DEFAULT_COLLECTION = "rag"  # 向量库里的集合名（Chroma collection / Qdrant collection）；RAG_COLLECTION 可改
VECTOR_STORES = ("chroma", "qdrant")

# 模型提供方：本地（sentence-transformers）或云端 API。
# embedding: local | openai（OpenAI 兼容的 POST {base}/embeddings）；reranker: local | api（POST {base}/rerank）| none（不精排）
EMBEDDING_PROVIDERS = ("local", "openai")
RERANKER_PROVIDERS = ("local", "api", "none")

HALF_PRECISION_DEVICES = ("mps", "cuda")  # 只有这些设备上用 fp16；CPU 上半精度又慢又不稳

LlmPurpose = Literal["answer", "eval"]  # answer: 生成回答；eval: 评测裁判 / ragas（可单独指定模型）


@dataclass(frozen=True)
class LLMConfig:
    """某个用途下实际生效的大模型参数（已合并 eval_llm_* 覆盖项）；api_key 是 SecretStr，repr 只显示掩码。"""

    protocol: str
    model: str | None
    base_url: str | None
    api_key: SecretStr | None
    temperature: float
    max_tokens: int | None
    timeout: float
    max_retries: int

    @property
    def configured(self) -> bool:
        """模型名、地址、密钥三项是否齐全。"""
        return bool(self.model and self.base_url and self.api_key)


@dataclass(frozen=True)
class ApiEndpoint:
    """某个云端模型接口（embedding / rerank）实际生效的连接参数（已合并对大模型配置的回退）；api_key 是 SecretStr，repr 只显示掩码。"""

    base_url: str | None
    api_key: SecretStr | None
    timeout: float
    max_retries: int
    base_url_from_llm: bool = False  # 地址是否回退到了 RAG_LLM_BASE_URL（仅用于提示）
    api_key_from_llm: bool = False  # 密钥是否回退到了 RAG_LLM_API_KEY


def _origin(url: str | None) -> str:
    """协议 + 主机 + 端口（小写），用来判断两个地址是不是同一个服务；解析不了返回空串。"""
    from urllib.parse import urlsplit

    try:
        parts = urlsplit((url or "").strip())
        return f"{parts.scheme}://{parts.netloc}".lower() if parts.scheme and parts.netloc else ""
    except ValueError:
        return ""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=None,  # .env 位置随 RAG_HOME 变化，在 __init__ 里按调用时的主目录指定
        env_file_encoding="utf-8",
        extra="ignore",  # .env 里还有 CHROMA_* 等别的变量
        populate_by_name=True,
    )

    # 模型提供方：本地（Hugging Face 仓库名）或云端 API（此时 *_model 是服务商的模型 ID）。
    # 换 embedding（模型 / 提供方 / 维度）需重建索引，换 reranker 需重新校准 min_score
    embedding_provider: Literal["local", "openai"] = "local"
    embedding_model: str = Field(DEFAULT_EMBEDDING_MODEL, min_length=1)
    reranker_provider: Literal["local", "api", "none"] = "local"
    reranker_model: str = Field(DEFAULT_RERANKER_MODEL, min_length=1)
    # 云端 embedding（OpenAI 兼容 POST {base}/embeddings）：地址 / 密钥留空则回退到 RAG_LLM_BASE_URL / RAG_LLM_API_KEY
    embedding_base_url: str | None = None
    embedding_api_key: SecretStr | None = None
    embedding_dimensions: int | None = Field(None, ge=1)  # 设置后作为 dimensions 参数发给服务端（支持 Matryoshka 的模型）；也计入索引签名
    embedding_batch_size: int = Field(16, ge=1)  # 每次请求最多发多少条文本（部分服务商有上限）
    embedding_query_instruction: str = ""  # 只加在查询前面的指令前缀（如 Qwen3 的 "Instruct: ...\nQuery: "）；文档不加；本地模型不用它
    embedding_timeout: float | None = Field(None, gt=0)  # 单次请求超时（秒）；留空沿用 RAG_LLM_TIMEOUT
    embedding_max_retries: int | None = Field(None, ge=0)  # 超时 / 连接错误 / 429 / 5xx 的重试次数；留空沿用 RAG_LLM_MAX_RETRIES
    # 云端 rerank（POST {base}/rerank，{model, query, documents, top_n}）：同样可回退到大模型的地址 / 密钥
    reranker_base_url: str | None = None
    reranker_api_key: SecretStr | None = None
    reranker_timeout: float | None = Field(None, gt=0)
    reranker_max_retries: int | None = Field(None, ge=0)
    # 本地模型设备（只对 local 提供方生效）：auto -> mps（Apple Silicon）> cuda > cpu
    device: Literal["auto", "mps", "cuda", "cpu"] = "auto"
    # 默认召回方式；API 可按请求覆盖
    mode: Literal["vector", "bm25", "hybrid"] = "hybrid"
    retrieve_k: int = Field(20, ge=1)  # 召回（精排前）候选数
    per_route_k: int = Field(30, ge=1)  # 混合检索每一路各取多少
    rerank_k: int = Field(5, ge=1)  # 精排后保留片段数
    min_score: float = Field(RERANK_MIN_SCORE, ge=0, le=1)  # 0 表示不拒答
    context_min_score: float = Field(CONTEXT_MIN_SCORE, ge=0, le=1)  # 0 表示不过滤
    log_level: str = "INFO"
    host: str = "127.0.0.1"
    port: int = Field(8000, ge=1, le=65535)
    # 接口鉴权：设置后所有 API 请求都要带这把共享密钥（Authorization: Bearer <key> 或 X-API-Key）；留空 = 不鉴权（本机使用）
    api_key: SecretStr | None = None
    # 允许跨域访问的前端来源，逗号分隔（如 https://app.example.com,http://localhost:5173）；留空 = 不加 CORS 中间件
    # NoDecode：环境变量按普通字符串交给下面的 validator 切分，而不是当 JSON 解析
    cors_origins: Annotated[list[str], NoDecode] = Field(default_factory=list)

    # 数据与索引位置（默认都在项目主目录 RAG_HOME 下，即运行目录里的 data/、chroma_db/ 等；测试里指向临时目录）
    data_dir: Path = Field(default_factory=lambda: project_home() / "data")  # 文档目录：POST /documents 上传到这里，POST /ingest 扫描这里
    db_dir: Path = Field(default_factory=lambda: project_home() / "chroma_db")  # Chroma 持久化目录（RAG_VECTOR_STORE=chroma 时用）
    meta_file: Path = Field(default_factory=lambda: project_home() / ".ingest_meta.json")  # 增量入库的文件指纹
    index_info_file: Path | None = None  # 索引的 embedding 签名（提供方 / 模型 / 维度）；留空 = 与 meta_file 同目录的 .index_info.json
    cache_dir: Path = Field(default_factory=lambda: project_home() / ".bm25_cache")  # BM25 分词缓存
    web_dist: Path = Field(default_factory=default_web_dist)  # 前端构建产物：默认 <主目录>/web/dist，没有则用随包的副本；相对路径按主目录解析，目录存在时由 API 服务托管
    max_upload_mb: int = Field(300, ge=1)  # POST /documents 的单文件大小上限（最大的教材 PDF 约 235 MB）

    # 向量库后端：chroma（默认，本地目录 db_dir）或 qdrant（本地嵌入式目录 qdrant_path，或服务端 qdrant_url）
    vector_store: Literal["chroma", "qdrant"] = "chroma"
    collection: str = Field(DEFAULT_COLLECTION, min_length=1)  # 集合名
    qdrant_path: Path = Field(default_factory=lambda: project_home() / "qdrant_db")  # Qdrant 本地模式的数据目录
    qdrant_url: str | None = None  # 设置后连接 Qdrant 服务端（如 http://localhost:6333），qdrant_path 被忽略
    qdrant_api_key: SecretStr | None = None  # Qdrant 服务端的 API Key（可选）

    # 大模型（默认 OpenAI 兼容协议）：RAG_LLM_*，未设置时回退到旧的 ARK_* 变量名（方舟，同为 OpenAI 兼容）。
    # 注意：validation_alias 会绕过 env_prefix，所以两个名字都要写全；AliasChoices 里靠前的别名优先。
    # pydantic-settings 按来源分层（环境变量整体优先于 .env），所以直接构造 Settings() 时 shell 里的 ARK_* 会压过 .env 里的 RAG_LLM_*；
    # 但命令行入口和 ragbase.ingest.common 会先 load_env()，把 .env 灌进 os.environ（不覆盖已有变量），
    # 此时两套名字都在环境变量这一层，RAG_LLM_* 总是优先于 ARK_*（见 docs/configuration.md“优先级”）。
    llm_protocol: Literal["openai"] = "openai"  # 新增协议（如 anthropic）：这里加字面量 + llm.factory 加一个分支
    llm_base_url: str | None = Field(None, validation_alias=AliasChoices("RAG_LLM_BASE_URL", "ARK_BASE_URL"))
    llm_api_key: SecretStr | None = Field(None, validation_alias=AliasChoices("RAG_LLM_API_KEY", "ARK_CHAT_API_KEY"))
    llm_model: str | None = Field(None, validation_alias=AliasChoices("RAG_LLM_MODEL", "ARK_CHAT_MODEL"))
    llm_temperature: float = Field(0, ge=0, le=2)
    llm_max_tokens: int | None = Field(None, ge=1)  # None 表示不限制，用服务端默认
    llm_timeout: float = Field(60, gt=0)  # 单次请求超时（秒）
    llm_max_retries: int = Field(2, ge=0)  # 超时 / 连接错误 / 429 / 5xx 的自动重试次数（指数退避）

    # 对话 Agent（POST /chat）：会话只存在进程内存里，重启即丢
    agent_max_tool_calls: int = Field(4, ge=1)  # 每轮用户消息最多调用几次工具（超出后工具被拒绝，模型只能据已有结果作答）
    agent_max_model_calls: int = Field(6, ge=2)  # 每轮用户消息最多调用几次大模型（超出后直接结束并返回提示）
    chat_history_turns: int = Field(10, ge=1)  # 每个会话保留最近几轮对话（一轮 = 一条用户消息及其全部回复 / 工具消息）
    chat_max_sessions: int = Field(200, ge=1)  # 内存中最多保留的会话数，超出时淘汰最久未使用的空闲会话
    chat_session_ttl: float = Field(3600, gt=0)  # 会话空闲多少秒后过期（秒）

    # 评测裁判（评测 runner 的 ragas 与中文裁判）可单独指定模型；留空则沿用上面的回答模型配置
    eval_llm_base_url: str | None = None
    eval_llm_api_key: SecretStr | None = None
    eval_llm_model: str | None = None

    def __init__(self, **values):
        values.setdefault("_env_file", env_file())  # 显式传 _env_file=None 仍可禁用 .env（测试用）
        super().__init__(**values)

    @field_validator("web_dist")
    @classmethod
    def _resolve_web_dist(cls, value: Path) -> Path:
        return value if value.is_absolute() else project_home() / value

    @field_validator("embedding_model", "reranker_model")
    @classmethod
    def _strip_model_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("模型名不能为空")
        return value

    @field_validator("embedding_provider", "reranker_provider", mode="before")
    @classmethod
    def _normalize_provider(cls, value):
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("embedding_base_url", "reranker_base_url", mode="before")
    @classmethod
    def _blank_api_url_to_none(cls, value):
        return (value.strip() or None) if isinstance(value, str) else value

    @field_validator("embedding_dimensions", "embedding_timeout", "embedding_max_retries", "reranker_timeout", "reranker_max_retries", "index_info_file", "llm_max_tokens", mode="before")
    @classmethod
    def _blank_number_to_none(cls, value):
        return (value.strip() or None) if isinstance(value, str) else value

    @field_validator("embedding_query_instruction")
    @classmethod
    def _unescape_instruction(cls, value: str) -> str:
        """环境变量 / .env 里很难写真正的换行，所以把两个字符 \\n 还原成换行（"Instruct: ...\\nQuery: "）。"""
        return value.replace("\\n", "\n")

    @field_validator("log_level")
    @classmethod
    def _normalize_log_level(cls, value: str) -> str:
        value = value.strip().upper()
        if value not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            raise ValueError(f"log_level 无效：{value!r}")
        return value

    @field_validator("vector_store", mode="before")
    @classmethod
    def _normalize_vector_store(cls, value):
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("qdrant_url", mode="before")
    @classmethod
    def _blank_url_to_none(cls, value):
        return (value.strip() or None) if isinstance(value, str) else value

    @field_validator("llm_model", "llm_base_url", "eval_llm_model", "eval_llm_base_url", mode="before")
    @classmethod
    def _blank_to_none(cls, value):
        return (value.strip() or None) if isinstance(value, str) else value

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_cors_origins(cls, value):
        if isinstance(value, str):
            value = value.split(",")
        return [item.strip().rstrip("/") for item in value if isinstance(item, str) and item.strip()]

    @field_validator(
        "api_key", "llm_api_key", "eval_llm_api_key", "qdrant_api_key", "embedding_api_key", "reranker_api_key", mode="before"
    )
    @classmethod
    def _blank_secret_to_none(cls, value):
        return (value.strip() or None) if isinstance(value, str) else value

    def llm_config(self, purpose: LlmPurpose = "answer") -> LLMConfig:
        """某用途下生效的大模型参数；eval 的模型 / 地址 / 密钥单独设置了就用它，没设置的沿用回答模型。"""
        override = purpose == "eval"
        return LLMConfig(
            protocol=self.llm_protocol,
            model=(self.eval_llm_model if override else None) or self.llm_model,
            base_url=(self.eval_llm_base_url if override else None) or self.llm_base_url,
            api_key=(self.eval_llm_api_key if override else None) or self.llm_api_key,
            temperature=self.llm_temperature,
            max_tokens=self.llm_max_tokens,
            timeout=self.llm_timeout,
            max_retries=self.llm_max_retries,
        )

    def _api_endpoint(self, base_url, api_key, timeout, max_retries) -> ApiEndpoint:
        """云端 embedding / rerank 的连接参数：地址未设置回退到大模型的地址；密钥未设置时，仅当最终地址与大模型是同一个服务（同协议 + 主机 + 端口）
        才回退到大模型的密钥——避免把 A 服务商的密钥发给显式配置的 B 服务；超时 / 重试未设置沿用大模型的。"""
        llm = self.llm_config("answer")
        url, url_from_llm = (base_url, False) if base_url else (llm.base_url, bool(llm.base_url))
        key, key_from_llm = api_key, False
        if key is None and llm.api_key is not None and _origin(url) and _origin(url) == _origin(llm.base_url):
            key, key_from_llm = llm.api_key, True
        return ApiEndpoint(
            base_url=url,
            api_key=key,
            timeout=timeout if timeout is not None else llm.timeout,
            max_retries=max_retries if max_retries is not None else llm.max_retries,
            base_url_from_llm=url_from_llm,
            api_key_from_llm=key_from_llm,
        )

    def embedding_endpoint(self) -> ApiEndpoint:
        return self._api_endpoint(
            self.embedding_base_url, self.embedding_api_key, self.embedding_timeout, self.embedding_max_retries
        )

    def reranker_endpoint(self) -> ApiEndpoint:
        return self._api_endpoint(
            self.reranker_base_url, self.reranker_api_key, self.reranker_timeout, self.reranker_max_retries
        )

    @property
    def index_info_path(self) -> Path:
        """索引 embedding 签名文件的位置：RAG_INDEX_INFO_FILE，缺省与 meta_file 同目录（默认即 <主目录>/.index_info.json）。"""
        return self.index_info_file or self.meta_file.with_name(".index_info.json")

    @property
    def uses_local_embedding(self) -> bool:
        return self.embedding_provider == "local"

    @property
    def uses_local_reranker(self) -> bool:
        return self.reranker_provider == "local"

    @property
    def uses_local_models(self) -> bool:
        """是否至少有一个本地模型（需要 torch / 设备 / 模型缓存）。"""
        return self.uses_local_embedding or self.uses_local_reranker

    @property
    def llm_configured(self) -> bool:
        """回答模型三项配置是否齐全（只返回布尔值，不暴露密钥）。"""
        return self.llm_config("answer").configured


@lru_cache
def get_settings() -> Settings:
    """进程内缓存的全局配置；测试里改了环境变量后用 get_settings.cache_clear() 重新读取。"""
    return Settings()


LOCAL_ML_MODULES = ("torch", "transformers", "sentence_transformers")


def block_local_ml_imports() -> list[str]:
    """把 torch / transformers / sentence_transformers 标记为“不可导入”（sys.modules[name] = None），返回被标记的模块名。

    用途：embedding 与 reranker 都用云端（或不精排）时，本项目自己的代码不需要它们，但 langchain-core 只要发现
    transformers 已安装就会在 import 时顺带加载 transformers + torch（约 1 秒、数百 MB 内存）。在导入 langchain 之前调用本函数，
    它的 try/except ImportError 就会跳过。已经导入过的模块不动。命令行入口只在全云端配置的 serve / ask / chat / ingest 下调用。
    """
    import sys

    blocked = []
    for name in LOCAL_ML_MODULES:
        if name not in sys.modules:
            sys.modules[name] = None  # type: ignore[assignment]
            blocked.append(name)
    return blocked


def resolve_device(device: str = "auto") -> str:
    """把 auto 解析成具体设备：mps > cuda > cpu；显式指定的设备原样返回。"""
    if device != "auto":
        return device
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
