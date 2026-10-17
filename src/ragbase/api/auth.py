"""静态 API Key 鉴权：整个产品只有一把共享密钥（RAG_API_KEY），没有用户体系。

设计：鉴权放在 app 的 HTTP 中间件里（见 app.py 的 request_context），按「默认拒绝 + 显式公开白名单」判定，
而不是给每个路由挂依赖——
- 新增路由默认就受保护，不会因为忘了加依赖而裸奔；公开什么只在 is_public() 一处声明；
- 在读取请求体之前就拒绝：FastAPI 的依赖在解析 multipart 之后才执行，未鉴权的人可以先让服务落盘数百 MB 上传；
- /chat/stream 在开始流式前就被拒绝。
密钥与请求头里的值从不写日志、不出现在响应里。
"""
import ipaddress
import re
import secrets
from collections.abc import Mapping

from pydantic import SecretStr

UNAUTHORIZED_DETAIL = "需要有效的 API Key：请在请求头带上 Authorization: Bearer <key>（或 X-API-Key: <key>）"
NO_AUTH_EXPOSED_WARNING = "未设置 RAG_API_KEY，服务对外暴露且没有鉴权"

# 公开路径（无需密钥）：前端静态资源与 SPA 路由、交互文档（schema 里没有密钥）。/health 单独处理（见 is_health）。
_PUBLIC_EXACT = frozenset({"/", "/library", "/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"})
_PUBLIC_PATTERNS = (re.compile(r"/c/[^/]+"), re.compile(r"/assets/.+"))
_PUBLIC_METHODS = frozenset({"GET", "HEAD"})


def is_public(method: str, path: str) -> bool:
    """该请求是否无需密钥。只放行 GET / HEAD 的前端页面、静态资源和文档；其余一律要密钥。"""
    if method.upper() not in _PUBLIC_METHODS:
        return False
    return path in _PUBLIC_EXACT or any(pattern.fullmatch(path) for pattern in _PUBLIC_PATTERNS)


def is_health(method: str, path: str) -> bool:
    """/health 对所有人开放，但没有有效密钥时只返回精简内容（由路由自己判断 request.state.authenticated）。"""
    return path == "/health" and method.upper() in _PUBLIC_METHODS


def extract_keys(headers: Mapping[str, str]) -> list[str]:
    """从请求头取出客户端出示的密钥：Authorization: Bearer <key> 和 X-API-Key: <key>（两种都带时两个都参与校验）。"""
    keys = []
    authorization = headers.get("authorization", "")
    scheme, _, credentials = authorization.partition(" ")
    if scheme.lower() == "bearer" and credentials.strip():
        keys.append(credentials.strip())
    explicit = headers.get("x-api-key", "").strip()
    if explicit:
        keys.append(explicit)
    return keys


def key_matches(provided: str, expected: SecretStr) -> bool:
    """常量时间比较。HTTP 头在 Starlette 里按 latin-1 解码，这里还原成原始字节再和配置里的 UTF-8 密钥比。"""
    try:
        provided_bytes = provided.encode("latin-1")
    except UnicodeEncodeError:
        return False
    return secrets.compare_digest(provided_bytes, expected.get_secret_value().encode("utf-8"))


def is_authenticated(headers: Mapping[str, str], expected: SecretStr | None) -> bool:
    """鉴权关闭（expected 为 None）视为已通过；否则出示的任一密钥匹配即通过。"""
    if expected is None:
        return True
    # 先算完所有比较再汇总（列表推导不短路），密钥最多两个
    return any([key_matches(key, expected) for key in extract_keys(headers)])


def is_loopback_host(host: str) -> bool:
    """绑定地址是否只对本机开放：localhost、127.0.0.0/8、::1。0.0.0.0 / :: / 具体网卡地址都算对外暴露。"""
    host = host.strip().strip("[]").lower()
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False
