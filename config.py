"""集中配置：环境变量 / .env -> Settings，命令行脚本与 API 服务共用，保证默认值只有一个来源。

- 自有配置统一加 RAG_ 前缀（如 RAG_DEVICE、RAG_PORT）；大模型沿用已有的 ARK_* 变量名。
- API Key 用 SecretStr 保存，repr / 日志里只会显示掩码；读取明文请用 get_secret_value()。
- 本模块不 import torch 等重型依赖（resolve_device 里才懒加载），可在任何脚本里放心导入。
"""
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).parent
ENV_FILE = BASE_DIR / ".env"

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

HALF_PRECISION_DEVICES = ("mps", "cuda")  # 只有这些设备上用 fp16；CPU 上半精度又慢又不稳


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",  # .env 里还有 CHROMA_* 等别的变量
        populate_by_name=True,
    )

    # 本地模型设备：auto -> mps（Apple Silicon）> cuda > cpu
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

    # 数据与索引位置（默认即仓库内的 data/、chroma_db/ 等；测试里指向临时目录）
    data_dir: Path = BASE_DIR / "data"  # 文档目录：POST /documents 上传到这里，POST /ingest 扫描这里
    db_dir: Path = BASE_DIR / "chroma_db"  # Chroma 持久化目录
    meta_file: Path = BASE_DIR / ".ingest_meta.json"  # 增量入库的文件指纹
    cache_dir: Path = BASE_DIR / ".bm25_cache"  # BM25 分词缓存
    max_upload_mb: int = Field(300, ge=1)  # POST /documents 的单文件大小上限（最大的教材 PDF 约 235 MB）

    # 回答模型（火山方舟，OpenAI 兼容接口），沿用原有 ARK_* 变量名
    chat_model: str | None = Field(None, validation_alias="ARK_CHAT_MODEL")
    chat_api_key: SecretStr | None = Field(None, validation_alias="ARK_CHAT_API_KEY")
    base_url: str | None = Field(None, validation_alias="ARK_BASE_URL")

    @field_validator("log_level")
    @classmethod
    def _normalize_log_level(cls, value: str) -> str:
        value = value.strip().upper()
        if value not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            raise ValueError(f"log_level 无效：{value!r}")
        return value

    @field_validator("chat_model", "base_url", mode="before")
    @classmethod
    def _blank_to_none(cls, value):
        return (value.strip() or None) if isinstance(value, str) else value

    @field_validator("chat_api_key", mode="before")
    @classmethod
    def _blank_secret_to_none(cls, value):
        return (value.strip() or None) if isinstance(value, str) else value

    @property
    def llm_configured(self) -> bool:
        """回答模型三项配置是否齐全（只返回布尔值，不暴露密钥）。"""
        return bool(self.chat_model and self.base_url and self.chat_api_key)


@lru_cache
def get_settings() -> Settings:
    """进程内缓存的全局配置；测试里改了环境变量后用 get_settings.cache_clear() 重新读取。"""
    return Settings()


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
