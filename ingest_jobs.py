"""入库任务注册表：同一时刻只允许一个任务（入库或删除文档），在后台线程里运行，状态保存在内存。

“忙”由一把不阻塞获取的锁表示：start() 与 exclusive() 都先抢它，抢不到就抛 JobRunningError（API 回 409）。
任务线程结束时（含 BM25 刷新）才释放，所以任务显示 succeeded 时索引和 BM25 都已是最新。
"""
import logging
import threading
import uuid
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

MAX_JOBS_KEPT = 100  # 内存里最多保留最近这么多个任务的状态


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class JobRunningError(RuntimeError):
    """已有任务在运行；running_job_id 可能为 None（忙于删除文档等非任务操作）。"""

    def __init__(self, running_job_id: str | None = None):
        super().__init__("已有入库任务在运行" + (f"（job_id={running_job_id}）" if running_job_id else ""))
        self.running_job_id = running_job_id


@dataclass
class IngestJob:
    job_id: str
    status: str = "pending"  # pending | running | succeeded | failed
    files_total: int = 0
    files_done: int = 0
    current_file: str | None = None
    chunks_added: int = 0
    chunks_deleted: int = 0
    skipped_files: list[str] = field(default_factory=list)
    error: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def update(self, **values) -> None:
        with self._lock:
            for key, value in values.items():
                setattr(self, key, value)

    def add(self, **increments) -> None:
        with self._lock:
            for key, value in increments.items():
                setattr(self, key, getattr(self, key) + value)

    def to_dict(self) -> dict:
        """线程安全的状态快照（不含内部锁）。"""
        with self._lock:
            data = {item.name: getattr(self, item.name) for item in fields(self) if not item.name.startswith("_")}
            data["skipped_files"] = list(self.skipped_files)
            return data


def progress_handler(job: IngestJob):
    """把 ingest_service 的进度事件映射到任务状态字段。"""

    def on_progress(event: str, **info) -> None:
        if event == "plan":
            job.update(files_total=len(info["to_add"]) + len(info["deleted"]), skipped_files=list(info["skipped"]))
        elif event == "no_documents":
            job.update(skipped_files=list(info["skipped"]))
        elif event == "migrate":
            job.add(chunks_deleted=info["chunks"])
        elif event == "file_start":
            job.update(current_file=info["name"])
        elif event == "file_done":
            job.add(files_done=1, chunks_added=info["chunks_added"], chunks_deleted=info["chunks_deleted"])
        elif event == "file_deleted":
            job.add(files_done=1, chunks_deleted=info["chunks_deleted"])
        elif event == "file_failed":
            job.add(files_done=1)

    return on_progress


class IngestJobManager:
    def __init__(self):
        self._busy = threading.Lock()
        self._jobs: OrderedDict[str, IngestJob] = OrderedDict()
        self._guard = threading.Lock()  # 保护 _jobs 与 _running_id
        self._running_id: str | None = None

    @property
    def running(self) -> bool:
        return self._busy.locked()

    def get(self, job_id: str) -> IngestJob | None:
        with self._guard:
            return self._jobs.get(job_id)

    def start(self, run) -> IngestJob:
        """在后台线程运行 run(job)。run 正常返回 -> succeeded；抛异常 -> failed（error 为异常描述）。"""
        if not self._busy.acquire(blocking=False):
            raise JobRunningError(self._running_id)
        job = IngestJob(job_id=uuid.uuid4().hex[:12])
        with self._guard:
            self._jobs[job.job_id] = job
            while len(self._jobs) > MAX_JOBS_KEPT:
                self._jobs.popitem(last=False)
            self._running_id = job.job_id
        threading.Thread(target=self._run, args=(job, run), name=f"ingest-{job.job_id}", daemon=True).start()
        return job

    def _run(self, job: IngestJob, run) -> None:
        job.update(status="running", started_at=utc_now())
        logger.info("入库任务开始：%s", job.job_id)
        status, error = "succeeded", None
        try:
            run(job)
        except Exception as exc:
            logger.exception("入库任务失败：%s", job.job_id)
            status, error = "failed", str(exc) or type(exc).__name__
        finally:
            with self._guard:
                self._running_id = None
            self._busy.release()
            # 先释放“忙”再写终态：客户端看到 succeeded / failed 时一定可以立刻启动下一个任务。
            # 状态、错误、结束时间一次写入，轮询方不会看到 succeeded 但 finished_at 还是空的中间态
            job.update(status=status, error=error, current_file=None, finished_at=utc_now())
            logger.info("入库任务结束：%s %s", job.job_id, status)

    @contextmanager
    def exclusive(self):
        """同步执行需要独占索引的操作（如删除文档）：期间新任务 / 其他独占操作得到 JobRunningError。"""
        if not self._busy.acquire(blocking=False):
            raise JobRunningError(self._running_id)
        try:
            yield
        finally:
            self._busy.release()
