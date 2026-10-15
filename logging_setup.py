"""日志配置：根 logger 输出 时间 / 级别 / logger 名 / 请求 ID / 消息。

请求 ID 通过 contextvar 传递（API 中间件设置），所以同一请求内任意模块的日志都能带上它。
"""
import contextvars
import logging

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s [%(request_id)s] %(message)s"


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


def setup_logging(level: str = "INFO") -> None:
    """配置根 logger（可重复调用，每次覆盖旧 handler）。"""
    handler = logging.StreamHandler()
    handler.addFilter(_RequestIdFilter())
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logging.basicConfig(level=level, handlers=[handler], force=True)
    # 访问日志由 API 中间件统一输出（含耗时和拒答信息），关掉 uvicorn 自带的一份避免重复
    logging.getLogger("uvicorn.access").disabled = True
    # httpx 会逐条打印大模型请求、bm25s 会打印建索引细节，噪音大且无用
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("bm25s").setLevel(logging.WARNING)
