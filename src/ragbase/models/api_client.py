"""云端模型接口（embedding / rerank）共用的 HTTP 客户端：超时、指数退避重试、不泄露密钥的错误信息。

只依赖 httpx（不导入 torch 等），所以全云端配置启动时不会加载任何本地模型库。
"""
import logging
import time

import httpx

from ragbase.config import ApiEndpoint

logger = logging.getLogger(__name__)

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
BACKOFF_BASE = 0.5  # 第 n 次重试前等待 BACKOFF_BASE * 2**n 秒；测试里可改小
BODY_SNIPPET_CHARS = 200


class ProviderError(RuntimeError):
    """调用云端 embedding / rerank 接口失败；消息可直接展示给用户，不含密钥与请求头。

    timeout 为 True 表示是超时（与 GenerationError 的约定一致）。
    """

    def __init__(self, message: str, timeout: bool = False):
        super().__init__(message)
        self.timeout = timeout


class ApiClient:
    """对一个 OpenAI 风格接口（{base}/{path}）发 POST JSON；密钥只放在请求头里，错误信息里会被抹掉。"""

    def __init__(self, endpoint: ApiEndpoint, label: str):
        if not endpoint.base_url:
            raise ValueError(f"{label} 未配置接口地址")
        self.label = label
        self.base_url = endpoint.base_url.rstrip("/")
        self.max_retries = endpoint.max_retries
        self._secret = endpoint.api_key.get_secret_value() if endpoint.api_key else ""
        headers = {"Content-Type": "application/json"}
        if self._secret:
            headers["Authorization"] = f"Bearer {self._secret}"
        self._client = httpx.Client(headers=headers, timeout=httpx.Timeout(endpoint.timeout))

    def __repr__(self) -> str:  # 不带密钥
        return f"ApiClient({self.label!r}, {self.base_url!r})"

    def _clean(self, text: str) -> str:
        return text.replace(self._secret, "***") if self._secret else text

    def post(self, path: str, payload: dict) -> dict:
        """POST {base}/{path}，返回解析后的 JSON 对象；超时 / 连接错误 / 429 / 5xx 自动重试，其余错误立即抛 ProviderError。"""
        url = f"{self.base_url}/{path.lstrip('/')}"
        last = "未知错误"
        timed_out = False
        for attempt in range(self.max_retries + 1):
            if attempt:
                time.sleep(BACKOFF_BASE * 2 ** (attempt - 1))
            try:
                response = self._client.post(url, json=payload)
            except httpx.TimeoutException as exc:
                last, timed_out = f"请求超时（{type(exc).__name__}）", True
                continue
            except httpx.TransportError as exc:
                last, timed_out = f"网络错误（{type(exc).__name__}）", False
                continue
            timed_out = False
            if response.status_code in RETRY_STATUS and attempt < self.max_retries:
                last = f"HTTP {response.status_code}"
                continue
            return self._parse(response, url, retried=attempt)
        raise ProviderError(f"{self.label} 接口调用失败：{last}（已重试 {self.max_retries} 次，地址 {self.base_url}）", timeout=timed_out)

    def _parse(self, response: httpx.Response, url: str, retried: int = 0) -> dict:
        if response.status_code >= 400:
            snippet = self._clean(response.text.strip().replace("\n", " "))[:BODY_SNIPPET_CHARS]
            hint = "（检查密钥是否正确）" if response.status_code in (401, 403) else "（检查地址 / 模型名是否正确）" if response.status_code == 404 else ""
            tried = f"（已重试 {retried} 次）" if retried else ""
            raise ProviderError(f"{self.label} 接口返回 HTTP {response.status_code}{hint}：{snippet}{tried}（地址 {url}）")
        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderError(f"{self.label} 接口返回的不是 JSON（地址 {url}，请确认地址以 /v1 之类的 API 根路径结尾）") from exc
        if not isinstance(data, dict):
            raise ProviderError(f"{self.label} 接口返回的 JSON 结构不符合预期（地址 {url}）")
        return data

    def close(self) -> None:
        self._client.close()
