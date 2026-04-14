"""User study data collection forms (SUS + task metrics).

Used for the 5-7 participant study: 3 scenarios each.
Collects: task completion, clicks, SUS score, free-form feedback.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

RESULTS_DIR = Path(__file__).resolve().parents[1] / "evaluation" / "results" / "user_study"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# Standard 10-item System Usability Scale
SUS_ITEMS = [
    "I think that I would like to use this system frequently.",
    "I found the system unnecessarily complex.",
    "I thought the system was easy to use.",
    "I think that I would need the support of a technical person to be able to use this system.",
    "I found the various functions in this system were well integrated.",
    "I thought there was too much inconsistency in this system.",
    "I would imagine that most people would learn to use this system very quickly.",
    "I found the system very cumbersome to use.",
    "I felt very confident using the system.",
    "I needed to learn a lot of things before I could get going with this system.",
]


@dataclass
class TaskRecord:
    scenario_id: str
    completed: bool
    task_time_sec: float
    clicks_agent: int
    clicks_manual: int  # Measured separately as baseline
    errors_encountered: int = 0


@dataclass
class StudySession:
    participant_id: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    task_records: List[TaskRecord] = field(default_factory=list)
    sus_responses: List[int] = field(default_factory=list)  # 1-5 for each of 10 items
    free_form_feedback: str = ""

    @property
    def sus_score(self) -> Optional[float]:
        """Standard SUS scoring: positive items (odd) - 1, negative items (even) 5 - response, sum * 2.5."""
        if len(self.sus_responses) != 10:
            return None
        total = 0
        for i, r in enumerate(self.sus_responses):
            if i % 2 == 0:  # Odd items (0-indexed even)
                total += r - 1
            else:
                total += 5 - r
        return total * 2.5

    @property
    def mean_clicks_saved(self) -> float:
        if not self.task_records:
            return 0.0
        return sum(max(0, t.clicks_manual - t.clicks_agent) for t in self.task_records) / len(self.task_records)

    @property
    def completion_rate(self) -> float:
        if not self.task_records:
            return 0.0
        return sum(1 for t in self.task_records if t.completed) / len(self.task_records)

    def save(self) -> Path:
        path = RESULTS_DIR / f"{self.participant_id}_{self.timestamp[:10]}.json"
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
        return path


def print_instructions() -> None:
    print("=" * 60)
    print("ClickLess AI User Study")
    print("=" * 60)
    print("""
Each participant completes 3 grocery shopping scenarios using ClickLess AI.
For each task, record:
  - Whether the task was completed successfully
  - Time taken (seconds)
  - Number of user clicks/interactions with the agent
  - Number of clicks estimated to complete same task manually on instacart.com

After all 3 tasks, administer the SUS questionnaire (1=Strongly Disagree, 5=Strongly Agree).

Target metrics:
  - Task completion rate >= 85%
  - SUS score >= 68 (above average)
  - Clicks saved >= 10 per session
""")
    print("=" * 60)


if __name__ == "__main__":
    print_instructions()
    for i, item in enumerate(SUS_ITEMS, 1):
        print(f"{i}. {item}")
