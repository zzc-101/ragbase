"""文档不漂移的检查（不加载模型、不访问网络）：

- docs/configuration.md 里由 scripts/gen_config_docs.py 生成的配置项表是最新的，且 Settings 的每个字段（含别名）都出现在文档里；
- 文档覆盖了每个命令行子命令和每个对外的 HTTP 路由；
- README.md、docs/*.md、CHANGELOG.md 里的相对链接（含 #锚点）都能解析。

运行：uv run python tests/test_docs.py
"""
import importlib.util
import re
from pathlib import Path

from ragbase import cli
from ragbase.config import Settings

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
CONFIG_DOC = DOCS / "configuration.md"
MARKDOWN_FILES = [ROOT / "README.md", ROOT / "CHANGELOG.md", *sorted(DOCS.glob("*.md"))]
EXPECTED_DOCS = {
    "configuration.md", "models.md", "api.md", "deployment.md", "architecture.md",
    "evaluation.md", "diagnostics.md", "development.md", "troubleshooting.md",
}


def _load_generator():
    spec = importlib.util.spec_from_file_location("gen_config_docs", ROOT / "scripts" / "gen_config_docs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _all_env_names() -> dict[str, list[str]]:
    """Settings 每个字段对应的全部环境变量名（含别名），从模型本身推导，新增字段后测试自动覆盖。"""
    generator = _load_generator()
    return {name: generator.env_names(name, info) for name, info in Settings.model_fields.items()}


# ---------------------------------------------------------------- 配置文档

def test_expected_doc_files_exist():
    missing = sorted(name for name in EXPECTED_DOCS if not (DOCS / name).is_file())
    assert not missing, f"缺少文档：{missing}"
    assert (ROOT / "README.md").is_file() and (ROOT / "CHANGELOG.md").is_file()


def test_every_settings_field_is_documented():
    text = CONFIG_DOC.read_text(encoding="utf-8")
    missing = [
        f"{field} -> {name}"
        for field, names in _all_env_names().items()
        for name in names
        if f"`{name}`" not in text
    ]
    assert not missing, f"docs/configuration.md 没有写出这些配置项（运行 scripts/gen_config_docs.py）：{missing}"
    assert "`RAG_HOME`" in text  # 不是 Settings 字段，但决定所有路径


def test_generated_block_is_up_to_date():
    generator = _load_generator()
    generator.check_coverage()  # 每个 Settings 字段都有分组和说明，且没有多余项
    assert generator.is_up_to_date(CONFIG_DOC), "docs/configuration.md 的配置项表已过期：运行 uv run python scripts/gen_config_docs.py"


def test_aliases_are_documented_as_legacy_names():
    text = CONFIG_DOC.read_text(encoding="utf-8")
    for legacy in ("ARK_BASE_URL", "ARK_CHAT_API_KEY", "ARK_CHAT_MODEL"):
        assert legacy in text, legacy


# ---------------------------------------------------------------- 命令行与接口覆盖

def test_every_cli_subcommand_is_mentioned():
    parser = cli.build_parser()
    subparsers = next(action for action in parser._actions if getattr(action, "choices", None) and "serve" in action.choices)
    corpus = "\n".join(path.read_text(encoding="utf-8") for path in MARKDOWN_FILES)
    missing = [name for name in subparsers.choices if f"ragbase {name}" not in corpus]
    assert not missing, f"文档里没有出现这些子命令：{missing}"


def test_every_http_route_is_in_api_doc():
    from ragbase.api import app as api

    app = api.create_app(pipeline_factory=lambda settings: None, settings=Settings(_env_file=None))
    normalize = lambda value: re.sub(r"\{[^}]*\}", "{}", value)  # noqa: E731  路径参数名不同不算差异
    text = normalize((DOCS / "api.md").read_text(encoding="utf-8"))
    missing = []
    for route in app.routes:
        path = getattr(route, "path", "")
        if not getattr(route, "include_in_schema", False) or path in ("/openapi.json", "/docs", "/redoc", "/docs/oauth2-redirect"):
            continue
        for method in sorted(getattr(route, "methods", ()) or ()):
            if method in ("HEAD", "OPTIONS"):
                continue
            if not re.search(rf"{method} {re.escape(normalize(path))}(?![\w/{{])", text):  # 后面不能紧跟更长的路径
                missing.append(f"{method} {path}")
    assert not missing, f"docs/api.md 没有写出这些接口（格式：METHOD /path）：{missing}"


# ---------------------------------------------------------------- 链接

_FENCE = re.compile(r"^\s*(```|~~~)")
_LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")


def _strip_code(text: str) -> str:
    """去掉围栏代码块和行内代码，避免把示例里的括号当成链接。"""
    kept, in_fence = [], False
    for line in text.splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence:
            kept.append(re.sub(r"`[^`]*`", "", line))
    return "\n".join(kept)


def _slug(heading: str) -> str:
    """GitHub 风格锚点：小写，去掉除字母 / 数字 / 空格 / 连字符 / 下划线外的字符，空格换成连字符。"""
    heading = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", heading)  # 链接只留文字
    heading = re.sub(r"[`*]", "", heading).strip().lower()
    heading = re.sub(r"[^\w\- ]", "", heading)
    return heading.replace(" ", "-")


def _anchors(path: Path) -> set[str]:
    seen: dict[str, int] = {}
    anchors = set()
    in_fence = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
            continue
        match = None if in_fence else re.match(r"^#{1,6}\s+(.*?)\s*#*\s*$", line)
        if match:
            slug = _slug(match.group(1))
            count = seen.get(slug, 0)
            seen[slug] = count + 1
            anchors.add(slug if count == 0 else f"{slug}-{count}")
    return anchors


def test_relative_links_resolve():
    problems = []
    for source in MARKDOWN_FILES:
        for target in _LINK.findall(_strip_code(source.read_text(encoding="utf-8"))):
            if re.match(r"^[a-z][a-z0-9+.-]*:", target, re.IGNORECASE):  # http(s):、mailto: 等
                continue
            path_part, _, fragment = target.partition("#")
            destination = source if not path_part else (source.parent / path_part).resolve()
            if not destination.exists():
                problems.append(f"{source.relative_to(ROOT)}: 链接目标不存在 {target}")
                continue
            if fragment and destination.suffix == ".md" and fragment.lower() not in _anchors(destination):
                problems.append(f"{source.relative_to(ROOT)}: 锚点不存在 {target}")
    assert not problems, "\n".join(problems)


def test_link_checker_itself_works():
    """自检：锚点算法对中英混排标题、全角括号、斜杠、行内代码的处理。"""
    assert _slug("按书过滤（`sources`）") == "按书过滤sources"
    assert _slug("GET /health") == "get-health"
    assert _slug("如何新增向量库 / 模型提供方 / LLM 协议") == "如何新增向量库--模型提供方--llm-协议"
    assert _slug("scripts/start.sh") == "scriptsstartsh"
    assert _strip_code("见 `[x](nope.md)` 和\n```\n[y](nope2.md)\n```\n[z](ok.md)").count("](") == 1


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"通过：{test.__name__}")
    print(f"共 {len(tests)} 项测试通过")
