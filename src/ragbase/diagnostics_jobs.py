"""体检任务注册表（Web 里的“开始体检”）：在后台线程里运行，状态保存在内存，轮询 GET /diagnose/jobs/{job_id} 查看。

与入库共用同一把 BusyLock（见 ragbase.ingest.jobs）：体检、入库、删除文档同一时刻只能有一个，冲突时 start() 抛 JobRunningError（API 回 409）。
本模块只用标准库，不导入模型相关代码。
"""
import logging
import threading
import uuid
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field, fields

from ragbase.ingest.jobs import MAX_JOBS_KEPT, BusyLock, utc_now

logger = logging.getLogger(__name__)


class RequestCounter:
    """线程安全的累计计数器：服务每收到一个对话请求加一，体检用“结束值 - 开始值”得到运行期间的对话请求数。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._value = 0

    def increment(self) -> None:
        with self._lock:
            self._value += 1

    @property
    def value(self) -> int:
        with self._lock:
            return self._value


@dataclass
class DiagnoseJob:
    job_id: str
    status: str = "pending"  # pending | running | succeeded | failed
    total: int = 0  # 本次体检的题数（去重、按上限截取之后）
    done: int = 0
    report_id: int | None = None  # 成功后报告在数据库里的 id
    error: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def update(self, **values) -> None:
        with self._lock:
            for key, value in values.items():
                setattr(self, key, value)

    def to_dict(self) -> dict:
        """线程安全的状态快照（不含内部锁）。"""
        with self._lock:
            return {item.name: getattr(self, item.name) for item in fields(self) if not item.name.startswith("_")}


class DiagnoseJobManager:
    def __init__(self, busy: BusyLock):
        self._busy = busy
        self._jobs: OrderedDict[str, DiagnoseJob] = OrderedDict()
        self._guard = threading.Lock()  # 保护 _jobs

    def get(self, job_id: str) -> DiagnoseJob | None:
        with self._guard:
            return self._jobs.get(job_id)

    def start(self, run: Callable[[DiagnoseJob], int], total: int = 0) -> DiagnoseJob:
        """在后台线程运行 run(job)，它返回报告 id。正常返回 -> succeeded；抛异常 -> failed（error 为中文说明）。
        total 是已知的题数，提前写进任务，让 202 响应就带上它。"""
        job = DiagnoseJob(job_id=uuid.uuid4().hex[:12], total=total)
        self._busy.acquire(job.job_id, "diagnose")  # 抢不到抛 JobRunningError
        with self._guard:
            self._jobs[job.job_id] = job
            while len(self._jobs) > MAX_JOBS_KEPT:
                self._jobs.popitem(last=False)
        threading.Thread(target=self._run, args=(job, run), name=f"diagnose-{job.job_id}", daemon=True).start()
        return job

    def _run(self, job: DiagnoseJob, run: Callable[[DiagnoseJob], int]) -> None:
        job.update(status="running", started_at=utc_now())
        logger.info("体检任务开始：%s", job.job_id)
        status, error, report_id = "succeeded", None, None
        try:
            report_id = run(job)
        except Exception as exc:
            logger.exception("体检任务失败：%s", job.job_id)
            status, error = "failed", f"体检失败：{str(exc) or type(exc).__name__}"
        finally:
            # 先释放“忙”再写终态：客户端看到 succeeded / failed 时一定可以立刻启动下一个任务
            self._busy.release()
            job.update(status=status, error=error, report_id=report_id, finished_at=utc_now())
            logger.info("体检任务结束：%s %s", job.job_id, status)
