#!/usr/bin/env python3
"""从 Settings 模型生成 docs/configuration.md 里的“配置项一览”表，保证文档不会和代码脱节。

- 环境变量名、类型 / 取值范围、默认值都直接从 ragbase.config.Settings 的字段推导；
  只有“说明”文字和分组写在本脚本的 GROUPS / DESCRIPTIONS 里。
- 新增 Settings 字段却忘了在这里补说明时，本脚本会报错（tests/test_docs.py 也会失败），所以不可能漏项。

用法（在仓库根目录）：
    uv run python scripts/gen_config_docs.py            # 更新 docs/configuration.md 里 BEGIN/END 标记之间的内容
    uv run python scripts/gen_config_docs.py --check    # 只检查是否最新（过期则退出码 1），不写文件
    uv run python scripts/gen_config_docs.py --print    # 把生成的 Markdown 打印到标准输出
"""
import argparse
import re
import sys
import types
from pathlib import Path
from typing import Literal, Union, get_args, get_origin

from pydantic import SecretStr
from pydantic_core import PydanticUndefined

from ragbase.config import Settings

ROOT = Path(__file__).resolve().parent.parent
DOC_PATH = ROOT / "docs" / "configuration.md"
BEGIN = "<!-- BEGIN GENERATED:settings -->"
END = "<!-- END GENERATED:settings -->"

# (分组标题, 分组说明, [字段名...])；顺序即文档顺序
GROUPS: list[tuple[str, str, list[str]]] = [
    (
        "路径与项目主目录",
        "所有运行时数据默认都放在**项目主目录**下：环境变量 `RAG_HOME`，未设置则为**当前工作目录**。"
        "`.env`、`bm25_userdict.txt`、`eval_questions.json`、`.ragas_cache/`、`eval_results.*.json` 的位置也由它决定（它们不是 Settings 字段，没有单独的变量）。\n\n"
        "`RAG_HOME` 必须是真正的环境变量（shell / Docker 的 `ENV`），**不能写在 `.env` 里**——`.env` 自己的位置就依赖它。"
        "下表的路径变量建议写绝对路径：除 `RAG_WEB_DIST` 外，相对路径不会按 `RAG_HOME` 解析（而是相对进程当前目录）。",
        ["data_dir", "db_dir", "meta_file", "index_info_file", "cache_dir"],
    ),
    (
        "检索与拒答",
        "这些值是服务和命令行的默认值；`/search`、`/ask` 的请求字段与 `ragbase ask` / `ragbase eval` 的命令行参数可按次覆盖。",
        ["mode", "retrieve_k", "per_route_k", "rerank_k", "min_score", "context_min_score"],
    ),
    (
        "向量库",
        "后端的取舍、迁移与本地 / 服务端模式的区别见 [architecture.md](architecture.md#向量库抽象) 与 [deployment.md](deployment.md)。",
        ["vector_store", "collection", "qdrant_path", "qdrant_url", "qdrant_api_key"],
    ),
    (
        "embedding",
        "提供方、示例组合与换模型的规则见 [models.md](models.md)。",
        [
            "embedding_provider", "embedding_model", "embedding_base_url", "embedding_api_key", "embedding_dimensions",
            "embedding_batch_size", "embedding_query_instruction", "embedding_timeout", "embedding_max_retries",
        ],
    ),
    (
        "reranker",
        "提供方与换 reranker 后需要重新标定阈值的规则见 [models.md](models.md)。",
        ["reranker_provider", "reranker_model", "reranker_base_url", "reranker_api_key", "reranker_timeout", "reranker_max_retries"],
    ),
    (
        "本地模型设备",
        "只对提供方为 `local` 的 embedding / reranker 生效；全云端时 `/health` 的 `device` 显示 `cloud`。",
        ["device"],
    ),
    (
        "大模型（回答与对话）",
        "走 OpenAI 兼容协议。`RAG_LLM_*` 是正式名，旧的 `ARK_*` 名字（火山方舟时代遗留）仍然兼容。三项（地址 / 密钥 / 模型名）缺任何一项都算“未配置”：服务仍可启动、`/search` 可用，`/ask` 与 `/chat` 返回 503。",
        [
            "llm_protocol", "llm_base_url", "llm_api_key", "llm_model", "llm_temperature", "llm_max_tokens",
            "llm_timeout", "llm_max_retries",
        ],
    ),
    (
        "评测裁判模型（可选）",
        "`ragbase eval` 的 Ragas 与中文裁判可以用另一个（通常更强的）模型；设置了哪项就覆盖哪项，其余沿用上面的回答模型。",
        ["eval_llm_base_url", "eval_llm_api_key", "eval_llm_model"],
    ),
    (
        "对话 Agent",
        "会话只存在服务进程内存里，重启即丢。机制见 [architecture.md](architecture.md#对话-agent)。",
        ["agent_max_tool_calls", "agent_max_model_calls", "chat_history_turns", "chat_max_sessions", "chat_session_ttl"],
    ),
    (
        "服务、鉴权与 CORS",
        "鉴权与暴露方式的完整说明见 [deployment.md](deployment.md#安全清单) 与 [api.md](api.md#鉴权)。",
        ["host", "port", "api_key", "cors_origins", "max_upload_mb", "log_level"],
    ),
    (
        "Web 界面",
        "",
        ["web_dist"],
    ),
]

# 字段 -> 说明（中文，单元格内不要出现竖线字符）
DESCRIPTIONS: dict[str, str] = {
    # 路径
    "data_dir": "文档目录。`POST /documents` 上传到这里，`ragbase ingest` / `POST /ingest` 扫描这里（只扫顶层，不递归）。",
    "db_dir": "Chroma 持久化目录（仅 `RAG_VECTOR_STORE=chroma` 时使用）。",
    "meta_file": "增量入库的文件指纹记录。",
    "index_info_file": "索引的 embedding 签名文件（提供方 / 模型 / 维度）。留空 = 与 `RAG_META_FILE` 同目录的 `.index_info.json`。",
    "cache_dir": "BM25 分词缓存目录。",
    # 检索
    "mode": "默认召回方式：`vector` 仅向量、`bm25` 仅关键词、`hybrid` 两路 RRF 融合。",
    "retrieve_k": "精排前的候选数（混合检索是 RRF 融合后保留的条数；单路模式是该路取的条数）。",
    "per_route_k": "混合检索时向量路和 BM25 路各取多少条参与融合。",
    "rerank_k": "精排后保留的片段数（`/search` 的 `top_k` 默认值，也是 `/ask`、对话 Agent 每次检索保留的条数）。",
    "min_score": "拒答阈值：最相关片段的精排得分低于它就拒答、**不调用大模型**；`0` 表示不拒答。"
                 "默认值按默认本地 reranker 标定，换 reranker 要重新标定；`RAG_RERANKER_PROVIDER=none` 时没有得分，不生效。",
    "context_min_score": "只把精排得分不低于它的片段交给大模型（得分最高的 top1 总是保留）；`0` 表示不过滤。得分已高度饱和，不要盲目调高。",
    # 向量库
    "vector_store": "向量库后端。取值不在列表内时启动报错。",
    "collection": "集合名（Chroma collection / Qdrant collection）。",
    "qdrant_path": "Qdrant **本地嵌入式**模式的数据目录（`RAG_QDRANT_URL` 未设置时使用）。同一目录同一时刻只允许一个进程打开。",
    "qdrant_url": "Qdrant **服务端**地址，如 `http://localhost:6333`（Docker Compose 里是 `http://qdrant:6333`）；设置后忽略 `RAG_QDRANT_PATH`。",
    "qdrant_api_key": "Qdrant 服务端的 API Key（可选；不会出现在日志 / `repr` 里）。",
    # embedding
    "embedding_provider": "`local`：sentence-transformers 加载本地模型；`openai`：OpenAI 兼容的 `POST {地址}/embeddings`。换提供方 / 模型 / 维度必须 `ragbase ingest --full` 重建索引。",
    "embedding_model": "本地：Hugging Face 仓库名；云端：服务商的模型 ID。",
    "embedding_base_url": "云端 embedding 的接口根地址（会请求 `{地址}/embeddings`，须以 `http://` 或 `https://` 开头）。留空回退到 `RAG_LLM_BASE_URL`。",
    "embedding_api_key": "云端 embedding 的密钥。留空时，仅当最终地址与 `RAG_LLM_BASE_URL` 是同一个服务（协议 + 主机 + 端口相同）才沿用 `RAG_LLM_API_KEY`，避免把 A 服务商的密钥发给 B 服务；本机 Ollama / vLLM 等不校验密钥的服务可以不填。",
    "embedding_dimensions": "可选：作为 `dimensions` 参数发给服务端（仅支持缩减维度的模型）；也计入索引签名。服务端返回的向量长度与它不符时入库会报错。本地模型忽略它。",
    "embedding_batch_size": "云端 embedding 每次请求最多发多少条文本；部分服务商有上限，请以服务商文档为准。",
    "embedding_query_instruction": "可选：只加在“查询”前面的指令前缀，文档不加（如 Qwen3-Embedding 的 `Instruct: ...\\nQuery: `，其中两个字符 `\\n` 会还原成换行）。本地模型自带 query prompt，不用它。",
    "embedding_timeout": "云端 embedding 单次请求超时（秒）。留空沿用 `RAG_LLM_TIMEOUT`。",
    "embedding_max_retries": "云端 embedding 超时 / 连接错误 / 429 / 5xx 的重试次数（指数退避）。留空沿用 `RAG_LLM_MAX_RETRIES`。",
    # reranker
    "reranker_provider": "`local`：CrossEncoder 加载本地模型；`api`：通用 `POST {地址}/rerank`；`none`：不精排（保持召回顺序，没有得分，**不会按相关度拒答**，只有召回为空才拒答）。",
    "reranker_model": "本地：Hugging Face 仓库名；云端：服务商的模型 ID。`none` 时不使用。",
    "reranker_base_url": "云端 rerank 的接口根地址（会请求 `{地址}/rerank`）。留空回退到 `RAG_LLM_BASE_URL`。",
    "reranker_api_key": "云端 rerank 的密钥。留空时的回退规则同 `RAG_EMBEDDING_API_KEY`。",
    "reranker_timeout": "云端 rerank 单次请求超时（秒）。留空沿用 `RAG_LLM_TIMEOUT`。",
    "reranker_max_retries": "云端 rerank 的重试次数。留空沿用 `RAG_LLM_MAX_RETRIES`。",
    # device
    "device": "本地模型设备：`auto` 依次尝试 mps（Apple Silicon）、cuda、cpu；mps / cuda 上用 fp16，cpu 保持默认精度。",
    # LLM
    "llm_protocol": "大模型协议。目前只实现 `openai`（OpenAI 兼容的 `/chat/completions`）；新增协议见 [development.md](development.md#如何新增向量库--模型提供方--llm-协议)。",
    "llm_base_url": "接口根地址，如 `https://api.openai.com/v1`。",
    "llm_api_key": "API Key（以 `SecretStr` 保存，不会出现在日志 / `repr` 里）。",
    "llm_model": "模型名。",
    "llm_temperature": "采样温度。",
    "llm_max_tokens": "单次回答的最大输出 token 数；不设置或留空 = 不限制，用服务端默认。",
    "llm_timeout": "单次请求超时（秒）。",
    "llm_max_retries": "超时 / 连接错误 / 429 / 5xx 的自动重试次数（指数退避），最坏等待约 `TIMEOUT × (RETRIES + 1)` 加退避间隔。",
    # eval
    "eval_llm_base_url": "评测裁判的接口地址；留空沿用回答模型的。",
    "eval_llm_api_key": "评测裁判的密钥；留空沿用回答模型的。",
    "eval_llm_model": "评测裁判的模型名；留空沿用回答模型的。",
    # agent
    "agent_max_tool_calls": "每轮用户消息最多调用几次工具；超出后工具被拒绝，模型只能据已有结果作答。",
    "agent_max_model_calls": "每轮用户消息最多调用几次大模型；超出后直接结束并返回提示“本轮处理步骤过多”。",
    "chat_history_turns": "每个会话保留最近几轮对话（一轮 = 一条用户消息及其全部回复 / 工具消息；整轮裁剪，不额外调用大模型做摘要）。",
    "chat_max_sessions": "内存中最多保留的会话数；超出时淘汰最久未使用的空闲会话（所有会话都忙时可暂时超出）。",
    "chat_session_ttl": "会话空闲多少秒后过期。",
    # service
    "host": "`ragbase serve` 的监听地址。`0.0.0.0` = 对外开放（务必同时设置 `RAG_API_KEY`）。",
    "port": "`ragbase serve` 的监听端口。",
    "api_key": "接口共享密钥：设置后所有 API 请求都要带它（`Authorization: Bearer <key>` 或 `X-API-Key`）；留空 = 不鉴权。",
    "cors_origins": "允许跨域访问的前端来源，逗号分隔（如 `https://app.example.com,http://localhost:5173`）；留空 = 不加 CORS 中间件。",
    "max_upload_mb": "`POST /documents` 的单文件大小上限（MB）；超限返回 413。",
    "log_level": "日志级别：`DEBUG` / `INFO` / `WARNING` / `ERROR` / `CRITICAL`（不区分大小写）。",
    # web
    "web_dist": "前端构建产物目录。默认 `<RAG_HOME>/web/dist`，没有则用随包分发的副本；相对路径按 `RAG_HOME` 解析。目录存在时由 API 服务在同一端口托管。",
}

# 默认值需要人工描述的字段（default_factory 算出来的路径，或“留空时有回退”的字段）
DEFAULT_TEXT: dict[str, str] = {
    "data_dir": "`<RAG_HOME>/data`",
    "db_dir": "`<RAG_HOME>/chroma_db`",
    "meta_file": "`<RAG_HOME>/.ingest_meta.json`",
    "index_info_file": "（空，即 `.index_info.json`，与 `RAG_META_FILE` 同目录）",
    "cache_dir": "`<RAG_HOME>/.bm25_cache`",
    "qdrant_path": "`<RAG_HOME>/qdrant_db`",
    "web_dist": "`<RAG_HOME>/web/dist`（没有则用随包副本）",
    "embedding_base_url": "（空，回退 `RAG_LLM_BASE_URL`）",
    "embedding_api_key": "（空，同源时回退 `RAG_LLM_API_KEY`）",
    "embedding_dimensions": "（空，不发送 `dimensions`）",
    "embedding_query_instruction": "（空）",
    "embedding_timeout": "（空，沿用 `RAG_LLM_TIMEOUT`）",
    "embedding_max_retries": "（空，沿用 `RAG_LLM_MAX_RETRIES`）",
    "reranker_base_url": "（空，回退 `RAG_LLM_BASE_URL`）",
    "reranker_api_key": "（空，同源时回退 `RAG_LLM_API_KEY`）",
    "reranker_timeout": "（空，沿用 `RAG_LLM_TIMEOUT`）",
    "reranker_max_retries": "（空，沿用 `RAG_LLM_MAX_RETRIES`）",
    "llm_max_tokens": "（不设置，不限制）",
    "eval_llm_base_url": "（空，沿用回答模型）",
    "eval_llm_api_key": "（空，沿用回答模型）",
    "eval_llm_model": "（空，沿用回答模型）",
    "api_key": "（空，不鉴权）",
    "cors_origins": "（空，不加 CORS）",
    "qdrant_url": "（空，用本地目录）",
    "qdrant_api_key": "（空）",
    "llm_base_url": "（空）",
    "llm_api_key": "（空）",
    "llm_model": "（空）",
}

# 带别名的字段：显示在“环境变量”列里的旧名说明
ALIAS_NOTE = {"ARK_": "旧名"}


def env_names(field_name: str, info) -> list[str]:
    """字段对应的环境变量名：带 validation_alias 的按别名顺序（优先的在前），否则 RAG_<字段名大写>。"""
    alias = info.validation_alias
    if alias is not None and hasattr(alias, "choices"):
        return [choice for choice in alias.choices if isinstance(choice, str)]
    return [f"RAG_{field_name.upper()}"]


def _unwrap_optional(annotation):
    if get_origin(annotation) in (Union, types.UnionType):
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


def type_text(info) -> str:
    """类型 / 取值范围：Literal 列出可选值，其余给出人话类型和 ge/le 等约束。"""
    annotation = _unwrap_optional(info.annotation)
    if get_origin(annotation) is Literal:
        return " / ".join(f"`{value}`" for value in get_args(annotation))
    if get_origin(annotation) is list:
        base = "逗号分隔列表"
    elif annotation is SecretStr:
        base = "密钥"
    elif annotation is Path:
        base = "路径"
    elif annotation is bool:
        base = "布尔"
    elif annotation is int:
        base = "整数"
    elif annotation is float:
        base = "数字"
    elif annotation is str:
        base = "字符串"
    else:
        base = getattr(annotation, "__name__", str(annotation))
    limits = []
    for meta in info.metadata:
        for attr, symbol in (("ge", "≥"), ("gt", ">"), ("le", "≤"), ("lt", "<")):
            value = getattr(meta, attr, None)
            if value is not None:
                limits.append(f"{symbol} {value:g}" if isinstance(value, float) else f"{symbol} {value}")
        if getattr(meta, "min_length", None):
            limits.append("非空")
    return base + (f"，{'，'.join(limits)}" if limits else "")


def default_text(field_name: str, info) -> str:
    if field_name in DEFAULT_TEXT:
        return DEFAULT_TEXT[field_name]
    if info.default_factory is not None:
        if info.default_factory is list:
            return "（空）"
        raise SystemExit(f"字段 {field_name} 的默认值来自 default_factory：请在 scripts/gen_config_docs.py 的 DEFAULT_TEXT 里补一条人工描述")
    value = info.default
    if value is PydanticUndefined:
        return "（必填）"
    if value is None:
        return "（空）"
    if isinstance(value, bool):
        return "`true`" if value else "`false`"
    if value == "":
        return "（空）"
    return f"`{value:g}`" if isinstance(value, float) else f"`{value}`"


def check_coverage() -> None:
    """Settings 的每个字段都必须出现在 GROUPS 且有说明；GROUPS / DESCRIPTIONS 里不能有已不存在的字段。"""
    fields = set(Settings.model_fields)
    grouped = [name for _, _, names in GROUPS for name in names]
    problems = []
    if missing := sorted(fields - set(grouped)):
        problems.append(f"这些 Settings 字段还没有放进任何分组（GROUPS）：{missing}")
    if extra := sorted(set(grouped) - fields):
        problems.append(f"GROUPS 里有 Settings 中不存在的字段：{extra}")
    if duplicated := sorted({name for name in grouped if grouped.count(name) > 1}):
        problems.append(f"这些字段出现在多个分组里：{duplicated}")
    if undocumented := sorted(fields - set(DESCRIPTIONS)):
        problems.append(f"这些 Settings 字段还没有说明（DESCRIPTIONS）：{undocumented}")
    if stale := sorted(set(DESCRIPTIONS) - fields):
        problems.append(f"DESCRIPTIONS 里有 Settings 中不存在的字段：{stale}")
    if problems:
        raise SystemExit("scripts/gen_config_docs.py 与 Settings 不一致：\n  " + "\n  ".join(problems))


def render() -> str:
    """生成 BEGIN / END 之间的 Markdown（不含标记行本身）。"""
    check_coverage()
    lines = ["> 本节由 `scripts/gen_config_docs.py` 从 `ragbase.config.Settings` 生成，请勿手改；改了 Settings 后运行 `uv run python scripts/gen_config_docs.py` 更新。", ""]
    for title, intro, names in GROUPS:
        lines.append(f"### {title}")
        lines.append("")
        if intro:
            lines += [intro, ""]
        lines += ["| 环境变量 | 类型 / 取值 | 默认值 | 说明 |", "| --- | --- | --- | --- |"]
        for name in names:
            info = Settings.model_fields[name]
            aliases = env_names(name, info)
            variable = f"`{aliases[0]}`" + "".join(f"<br>（旧名 `{alias}`）" for alias in aliases[1:])
            lines.append(f"| {variable} | {type_text(info)} | {default_text(name, info)} | {DESCRIPTIONS[name]} |")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _split(text: str) -> tuple[str, str, str]:
    start, end = text.find(BEGIN), text.find(END)
    if start == -1 or end == -1 or end < start:
        raise SystemExit(f"{DOC_PATH} 里找不到成对的标记：\n  {BEGIN}\n  {END}")
    return text[: start + len(BEGIN)], text[start + len(BEGIN): end], text[end:]


def current_block(path: Path = DOC_PATH) -> str:
    """文档里现有的生成区内容（BEGIN / END 之间，去掉首尾空行后的原样文本）。"""
    return _split(path.read_text(encoding="utf-8"))[1].strip("\n") + "\n"


def is_up_to_date(path: Path = DOC_PATH) -> bool:
    return current_block(path) == render()


def update(path: Path = DOC_PATH) -> bool:
    """更新生成区，返回文件内容是否有变化。"""
    text = path.read_text(encoding="utf-8")
    head, _, tail = _split(text)
    new_text = f"{head}\n\n{render()}\n{tail}"
    if new_text == text:
        return False
    path.write_text(new_text, encoding="utf-8")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="从 Settings 生成 docs/configuration.md 的配置项表")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true", help="只检查文档是否最新（过期则退出码 1）")
    group.add_argument("--print", dest="print_only", action="store_true", help="把生成的 Markdown 打印到标准输出，不改文件")
    args = parser.parse_args(argv)
    if args.print_only:
        sys.stdout.write(render())
        return 0
    if args.check:
        if is_up_to_date():
            print(f"{DOC_PATH.relative_to(ROOT)} 已是最新")
            return 0
        print(f"{DOC_PATH.relative_to(ROOT)} 已过期：请运行 uv run python scripts/gen_config_docs.py", file=sys.stderr)
        return 1
    changed = update()
    print(f"{DOC_PATH.relative_to(ROOT)} {'已更新' if changed else '已是最新，无需改动'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
