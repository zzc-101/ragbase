"""ragbase init：生成项目主目录的初始文件（data/、.env、bm25_userdict.txt）。

模板随包分发（src/ragbase/resources/），所以 pip / uv 安装后的包、Docker 镜像里同样可用，不依赖仓库目录。
默认绝不覆盖已有文件；--force 才覆盖，且会先把旧文件备份成 <名字>.bak。
"""
import os
import re
import secrets
import shutil
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

API_KEY_LINE = re.compile(r"^# RAG_API_KEY=.*$", re.MULTILINE)


@dataclass(frozen=True)
class InitAction:
    path: Path
    status: str  # created / skipped / overwritten
    note: str = ""


def read_resource(name: str) -> str:
    """读取随包分发的资源文件（UTF-8 文本）。"""
    return resources.files("ragbase").joinpath("resources", name).read_text(encoding="utf-8")


def generate_api_key() -> str:
    return secrets.token_urlsafe(32)


def render_env(api_key: str | None = None) -> str:
    """.env 内容：模板原样复制；给了 api_key 就把模板里的 `# RAG_API_KEY=` 占位行换成真实的一行。"""
    text = read_resource("env.example")
    if api_key:
        text, count = API_KEY_LINE.subn(lambda _m: f"RAG_API_KEY={api_key}", text, count=1)
        if count != 1:  # 模板被改坏了：宁可追加也不要悄悄丢掉密钥
            text += f"\nRAG_API_KEY={api_key}\n"
    return text


def _write(path: Path, content: str, force: bool, mode: int | None = None) -> InitAction:
    existed = path.exists()
    if existed and not force:
        return InitAction(path, "skipped", "已存在，保留不动")
    note = ""
    if existed:
        backup = path.with_name(path.name + ".bak")
        shutil.copy2(path, backup)
        note = f"旧文件已备份到 {backup.name}"
    path.write_text(content, encoding="utf-8")
    if mode is not None:
        os.chmod(path, mode)
    return InitAction(path, "overwritten" if existed else "created", note)


def init_home(home: Path, force: bool = False, api_key: str | None = None) -> list[InitAction]:
    """在 home 下创建 data/（含 README.txt）、.env、bm25_userdict.txt；返回每一项的处理结果。

    api_key 只在 .env 是这次新写入（新建或 --force 覆盖）时才生效；已有 .env 不动，所以不会出现“以为写入了其实没写”的假象——
    调用方可据返回的 .env 状态判断。
    """
    home = Path(home).expanduser().resolve()
    home.mkdir(parents=True, exist_ok=True)
    actions: list[InitAction] = []

    data_dir = home / "data"
    data_existed = data_dir.is_dir()
    data_dir.mkdir(exist_ok=True)
    actions.append(InitAction(data_dir, "skipped" if data_existed else "created", "文档目录" if not data_existed else "已存在"))
    actions.append(_write(data_dir / "README.txt", read_resource("data_readme.txt"), force))
    actions.append(_write(home / ".env", render_env(api_key), force, mode=0o600))  # 含密钥，仅本人可读
    actions.append(_write(home / "bm25_userdict.txt", read_resource("bm25_userdict.txt"), force))
    return actions
