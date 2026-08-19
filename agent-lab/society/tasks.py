"""Task definitions for the society experiment."""

from __future__ import annotations

from typing import List, Dict
import numpy as np


# Information fragment templates
CATEGORIES = ["technical", "social", "economic", "environmental", "political"]

TECHNICAL_FRAGMENTS = [
    "Open source code review reduces bugs by 40%",
    "TypeScript catches 15% more errors than JavaScript",
    "Unit test coverage above 80% reduces regression bugs",
    "Code refactoring improves developer productivity by 25%",
    "Pair programming doubles code quality metrics",
    "CI/CD pipelines reduce deployment failures by 60%",
    "API documentation reduces integration time by 35%",
    "Database indexing improves query performance by 10x",
    "Memory profiling reveals hidden leaks in 30% of apps",
    "Container deployment reduces environment issues by 70%",
]

SOCIAL_FRAGMENTS = [
    "Diverse teams produce 19% more innovative solutions",
    "Psychological safety increases team productivity by 12%",
    "Remote work increases output by 13% but reduces creativity",
    "Regular retrospectives improve team velocity by 15%",
    "Cross-functional teams solve problems 25% faster",
    "Open communication reduces project delays by 40%",
    "Team bonding activities improve collaboration by 20%",
    "Mentorship programs reduce new hire turnover by 25%",
    "Asynchronous communication improves deep work by 30%",
    "Flat hierarchies increase innovation by 22%",
]

ECONOMIC_FRAGMENTS = [
    "Agile methodology reduces time-to-market by 30%",
    "Technical debt costs 2-3x more if not addressed early",
    "Automation saves 20% of development time on average",
    "Cloud migration reduces infrastructure costs by 35%",
    "Open source adoption reduces software costs by 40%",
    "DevOps practices reduce operational costs by 25%",
    "Microservices architecture increases deployment frequency by 50%",
    "Code reuse reduces development costs by 30%",
    "Performance optimization improves user retention by 15%",
    "Security investment reduces breach costs by 50%",
]

ALL_FRAGMENTS = TECHNICAL_FRAGMENTS + SOCIAL_FRAGMENTS + ECONOMIC_FRAGMENTS


def generate_task(seed: int, task_num: int, n_agents: int = 128) -> dict:
    """Generate a task with information fragments."""
    rng = np.random.default_rng(seed * 100 + task_num)

    # Select 20-30 random fragments
    n_fragments = rng.integers(20, 31)
    selected_indices = rng.choice(len(ALL_FRAGMENTS), size=n_fragments, replace=False)
    fragments = [ALL_FRAGMENTS[i] for i in selected_indices]

    option_count = 8
    option_quality = rng.uniform(-1.0, 1.0, size=option_count)

    # Assign noisy evidence to alternatives. Reliability and latent quality are
    # retained by the environment but are not exposed to agent observations.
    information_fragments = []
    for i, frag in enumerate(fragments):
        option_id = i % option_count
        accuracy = float(rng.uniform(0.2, 0.95))
        noise = rng.normal(0, 1.0 - accuracy)
        info = {
            "content": f"Option {option_id}: {frag}",
            "category": CATEGORIES[i % len(CATEGORIES)],
            "accuracy": accuracy,
            "option_id": option_id,
            "value": float(option_quality[option_id] + noise),
            "id": f"frag_{task_num}_{i}",
        }
        information_fragments.append(info)

    # Generate task goal (synthetic)
    goals = [
        "Decide on a project management approach",
        "Choose a technology stack for the next quarter",
        "Determine team restructuring strategy",
        "Set priorities for resource allocation",
        "Evaluate and select a new tool/framework",
    ]

    task = {
        "id": task_num,
        "goal": goals[task_num % len(goals)],
        "information_fragments": information_fragments,
        "options": [f"option_{i}" for i in range(option_count)],
        "option_quality": option_quality.tolist(),
        "max_rounds": 50,
        "context": f"Task {task_num}: {goals[task_num % len(goals)]}",
    }

    return task


def generate_task_sequence(seed: int, n_tasks: int, n_agents: int = 128) -> List[dict]:
    """Generate a sequence of tasks for persistent society."""
    return [generate_task(seed, i, n_agents) for i in range(n_tasks)]
