"""依次运行 tests/test_*.py（每个脚本一个子进程，互不污染），打印各自的汇总行，任何一个失败则以非零码退出。

本地与 CI 共用：uv run python tests/run_all.py [名称片段 ...]
带名称片段时只跑文件名包含任一片段的脚本，如 tests/run_all.py llm api。
"""
import re
import subprocess
import sys
import time
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
ROOT = TESTS_DIR.parent
SUMMARY = re.compile(r"共 (\d+) 项测试通过")
TIMEOUT_SECONDS = 600  # 单个脚本的上限，防止某个测试卡死拖住整个 CI
TAIL_LINES = 40  # 失败时回显的输出行数


def discover(filters: list[str]) -> list[Path]:
    scripts = sorted(TESTS_DIR.glob("test_*.py"))
    return [path for path in scripts if not filters or any(part in path.name for part in filters)]


def run_script(path: Path) -> tuple[bool, str, float]:
    """运行一个脚本，返回 (是否通过, 汇总行或失败原因, 耗时秒)。失败时把输出尾部打印出来。"""
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            [sys.executable, str(path)],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return False, f"超过 {TIMEOUT_SECONDS}s 未结束", time.perf_counter() - start
    elapsed = time.perf_counter() - start
    output = completed.stdout + completed.stderr
    summary = next((line for line in reversed(output.splitlines()) if SUMMARY.search(line)), None)
    if completed.returncode == 0 and summary:
        return True, summary.strip(), elapsed
    print(f"----- {path.name} 输出（最后 {TAIL_LINES} 行）-----")
    print("\n".join(output.splitlines()[-TAIL_LINES:]))
    print("-" * 40)
    reason = f"退出码 {completed.returncode}" if completed.returncode else "没有输出汇总行"
    return False, reason, elapsed


def main(argv: list[str]) -> int:
    scripts = discover(argv)
    if not scripts:
        print("没有匹配的测试脚本")
        return 1
    failed, total = [], 0
    for path in scripts:
        ok, message, elapsed = run_script(path)
        print(f"[{'通过' if ok else '失败'}] {path.name:<26} {message}  ({elapsed:.1f}s)", flush=True)
        if ok:
            total += int(SUMMARY.search(message).group(1))
        else:
            failed.append(path.name)
    if failed:
        print(f"\n{len(failed)}/{len(scripts)} 个脚本失败：{', '.join(failed)}")
        return 1
    print(f"\n全部通过：{len(scripts)} 个脚本，共 {total} 项测试")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
