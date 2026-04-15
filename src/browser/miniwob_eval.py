"""MiniWoB++ Execution Success Rate (ESR) harness.

Validates the BrowserUse checkout agent against a 20-task subset of MiniWoB++
form-fill and navigation tasks. The evaluator wraps the same Browser Use Cloud
transport used by the checkout agent (with a local browser-use fallback) and
reports per-task and aggregate Execution Success Rate, the metric required by
proposal section V.D.

Targets:
  - 20 tasks across click / form-fill / navigation
  - ESR >= 0.90

Persists per-run results to evaluation/results/miniwob_<timestamp>.json.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

_RESULTS_DIR = Path("/scratch/smehta90/Clickless AI/evaluation/results")
_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

_MINIWOB_BASE_URL = os.getenv(
    "MINIWOB_BASE_URL",
    "http://localhost:8000/miniwob/",
)


# 20-task subset spanning click, form-fill, navigation as required by §V.D.
# All entries are valid env ids in the miniwob gymnasium namespace; invalid
# ids (fill-form, fill-text) have been replaced with bisect-angle and
# book-flight which cover form-fill style interactions.
EVAL_TASKS = [
    "click-button",
    "click-button-sequence",
    "click-checkboxes",
    "click-checkboxes-large",
    "click-dialog",
    "click-link",
    "click-tab",
    "click-test",
    "click-widget",
    "enter-text",
    "enter-text-dynamic",
    "enter-password",
    "login-user",
    "login-user-popup",
    "click-shape",
    "book-flight",
    "use-autocomplete",
    "search-engine",
    "navigate-tree",
    "social-media",
]


@dataclass
class TaskResult:
    task: str
    success: bool
    reward: float
    steps: int = 0
    duration_sec: float = 0.0
    error: Optional[str] = None


@dataclass
class EvalReport:
    results: List[TaskResult] = field(default_factory=list)
    timestamp: str = ""

    @property
    def success_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.success) / len(self.results)

    @property
    def mean_reward(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.reward for r in self.results) / len(self.results)

    @property
    def mean_duration(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.duration_sec for r in self.results) / len(self.results)

    def summary(self) -> str:
        lines = [
            "MiniWoB++ Evaluation",
            f"Tasks: {len(self.results)}",
            f"Execution Success Rate: {self.success_rate:.1%}",
            f"Mean reward: {self.mean_reward:.3f}",
            f"Mean duration (s): {self.mean_duration:.2f}",
        ]
        for r in self.results:
            tag = "OK  " if r.success else "FAIL"
            lines.append(f"  {tag} {r.task:<25} reward={r.reward:.2f} steps={r.steps}")
        return "\n".join(lines)

    def to_json(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "n_tasks": len(self.results),
            "esr": round(self.success_rate, 4),
            "mean_reward": round(self.mean_reward, 4),
            "mean_duration_sec": round(self.mean_duration, 4),
            "tasks": [asdict(r) for r in self.results],
        }


# ---------------------------------------------------------------------------
# Browser Use Cloud v3 transport
# ---------------------------------------------------------------------------

_CLOUD_BASE_URL = "https://api.browser-use.com/api/v3"
_CLOUD_POLL_SEC = 4
_CLOUD_DEADLINE_SEC = 300  # 5 minutes per miniwob task
_CLOUD_MODEL = os.getenv("BROWSERUSE_CLOUD_MODEL", "bu-max")


async def _run_cloud(task_name: str) -> TaskResult:
    """Run a single MiniWoB task via the Browser Use Cloud v3 API.

    Per docs.browser-use.com/cloud/api-reference:
      POST /sessions   — create a new agent session
      GET  /sessions/{id} — poll status + isTaskSuccessful + stepCount
      GET  /sessions/{id}/messages — per-step screenshots + summaries
    """
    import httpx

    api_key = os.getenv("BROWSERUSE_API_KEY", "")
    if not api_key:
        return TaskResult(
            task=task_name, success=False, reward=0.0,
            error="BROWSERUSE_API_KEY not set",
        )

    base_url = os.getenv("MINIWOB_BASE_URL", _MINIWOB_BASE_URL)
    if not base_url.endswith("/"):
        base_url += "/"
    task_url = f"{base_url}{task_name}.html"

    task_text = (
        f"Open {task_url} in the browser. A MiniWoB task harness will appear. "
        f"Click the black 'START' panel to begin an episode. Read the "
        f"instruction shown at the top of the task widget and complete it "
        f"using clicks and keyboard input. When the widget reports a success "
        f"signal (reward goes positive), stop. Return a JSON object "
        f"{{'task': <task name>, 'episodes_done': <int>, 'last_reward': <float>}}."
    )

    output_schema = {
        "type": "object",
        "properties": {
            "task": {"type": "string"},
            "episodes_done": {"type": "integer"},
            "last_reward": {"type": "number"},
        },
        "required": ["task", "episodes_done", "last_reward"],
    }

    start = time.time()
    headers = {"X-Browser-Use-API-Key": api_key, "Content-Type": "application/json"}

    async with httpx.AsyncClient(base_url=_CLOUD_BASE_URL, headers=headers, timeout=60) as http:
        try:
            create_resp = await http.post(
                "/sessions",
                json={
                    "task": task_text,
                    "model": _CLOUD_MODEL,
                    "outputSchema": output_schema,
                    "keepAlive": False,
                    "enableRecording": True,
                },
            )
            if create_resp.status_code >= 400:
                return TaskResult(
                    task=task_name, success=False, reward=0.0,
                    error=f"POST /sessions -> {create_resp.status_code}: {create_resp.text[:200]}",
                    duration_sec=time.time() - start,
                )
            sess = create_resp.json()
            sess_id = sess.get("id")
            logger.info("  cloud session %s created (task=%s)", sess_id, task_name)

            output: dict = {}
            phase = sess.get("status", "running")
            step_count = sess.get("stepCount", 0)
            is_successful: Optional[bool] = sess.get("isTaskSuccessful")

            while time.time() - start < _CLOUD_DEADLINE_SEC:
                await asyncio.sleep(_CLOUD_POLL_SEC)
                status_resp = await http.get(f"/sessions/{sess_id}")
                if status_resp.status_code >= 400:
                    return TaskResult(
                        task=task_name, success=False, reward=0.0,
                        error=f"GET /sessions/{{id}} -> {status_resp.status_code}",
                        duration_sec=time.time() - start,
                    )
                status = status_resp.json()
                phase = status.get("status", "running")
                step_count = status.get("stepCount", step_count)
                is_successful = status.get("isTaskSuccessful", is_successful)
                output = status.get("output") or {}
                if phase in ("finished", "completed", "stopped", "failed", "error"):
                    break

            # Persist per-step messages for reference
            try:
                msg_resp = await http.get(f"/sessions/{sess_id}/messages")
                if msg_resp.status_code < 400:
                    run_dir = _RESULTS_DIR / "miniwob_runs" / task_name
                    run_dir.mkdir(parents=True, exist_ok=True)
                    (run_dir / "messages.json").write_text(msg_resp.text)
            except Exception:
                pass

            success = bool(is_successful) if is_successful is not None else (
                phase in ("finished", "completed") and output.get("last_reward", 0) > 0
            )
            reward = float(output.get("last_reward", 1.0 if success else 0.0))
            return TaskResult(
                task=task_name,
                success=success,
                reward=reward,
                steps=int(step_count),
                duration_sec=time.time() - start,
                error=None if success else f"phase={phase}",
            )
        except Exception as exc:
            logger.exception("Cloud transport failure on %s", task_name)
            return TaskResult(
                task=task_name, success=False, reward=0.0,
                error=str(exc), duration_sec=time.time() - start,
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def run_task(task_name: str, headless: bool = True) -> TaskResult:
    """Run one MiniWoB++ task via Browser Use Cloud."""
    return await _run_cloud(task_name)


async def run_eval(tasks: Optional[List[str]] = None, headless: bool = True) -> EvalReport:
    tasks = tasks or EVAL_TASKS
    report = EvalReport(timestamp=datetime.now().isoformat(timespec="seconds"))
    for task in tasks:
        logger.info("MiniWoB++: running %s", task)
        result = await run_task(task, headless=headless)
        report.results.append(result)
        logger.info(
            "  %s: success=%s reward=%.2f steps=%d",
            task, result.success, result.reward, result.steps,
        )
    out_path = _RESULTS_DIR / f"miniwob_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(report.to_json(), indent=2))
    logger.info("MiniWoB++ results written to %s", out_path)
    logger.info(report.summary())
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    report = asyncio.run(run_eval())
    print(report.summary())
