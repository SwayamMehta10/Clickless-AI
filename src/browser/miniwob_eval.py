"""MiniWoB++ evaluation harness for the browser checkout agent.

Validates the agent on form-fill and navigation tasks before live checkout.
Target: >= 90% task success rate (ESR).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

# MiniWoB++ tasks relevant to grocery checkout
EVAL_TASKS = [
    "click-button",
    "click-checkboxes",
    "click-dialog",
    "enter-text",
    "login-user",
    "search-engine",
    "use-autocomplete",
    "fill-form",
    "navigate-tree",
]


@dataclass
class TaskResult:
    task: str
    success: bool
    reward: float
    error: Optional[str] = None
    steps: int = 0


@dataclass
class EvalReport:
    results: List[TaskResult] = field(default_factory=list)

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

    def summary(self) -> str:
        return (
            f"MiniWoB++ Evaluation\n"
            f"Tasks: {len(self.results)}\n"
            f"Success Rate: {self.success_rate:.1%}\n"
            f"Mean Reward: {self.mean_reward:.3f}\n"
            + "\n".join(
                f"  {'OK' if r.success else 'FAIL'} {r.task} (reward={r.reward:.2f})"
                for r in self.results
            )
        )


async def run_task(task_name: str, headless: bool = True) -> TaskResult:
    """Run a single MiniWoB++ task and return the result."""
    try:
        import gymnasium as gym
        import miniwob

        env = gym.make(
            f"miniwob/{task_name}-v1",
            render_mode=None if headless else "human",
        )
        obs, info = env.reset()

        # Simple rule-based agent for basic tasks
        total_reward = 0.0
        done = False
        steps = 0

        while not done and steps < 20:
            # Placeholder: random action (replace with actual agent)
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            steps += 1

        env.close()
        success = total_reward > 0.5
        return TaskResult(task=task_name, success=success, reward=total_reward, steps=steps)

    except ImportError:
        logger.warning("miniwob/gymnasium not installed. Skipping task: %s", task_name)
        return TaskResult(task=task_name, success=False, reward=0.0, error="not_installed")
    except Exception as exc:
        logger.error("Task %s failed: %s", task_name, exc)
        return TaskResult(task=task_name, success=False, reward=0.0, error=str(exc))


async def run_eval(tasks: Optional[List[str]] = None, headless: bool = True) -> EvalReport:
    """Run evaluation on all (or specified) MiniWoB++ tasks."""
    tasks = tasks or EVAL_TASKS
    report = EvalReport()

    for task in tasks:
        logger.info("Running MiniWoB++ task: %s", task)
        result = await run_task(task, headless=headless)
        report.results.append(result)
        logger.info("  %s: success=%s, reward=%.2f", task, result.success, result.reward)

    logger.info(report.summary())
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    report = asyncio.run(run_eval())
    print(report.summary())
