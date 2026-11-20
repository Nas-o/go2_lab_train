"""List the Go2 environments registered by this project."""

from mjlab.tasks.registry import list_tasks

import go2_rl_lab.tasks  # noqa: F401  (populates the task registry)


def discover_task_ids() -> list[str]:
    """Return the project task IDs registered with mjlab."""
    return [task_id for task_id in list_tasks() if task_id.startswith("go2-")]


def main() -> None:
    """Print every registered Go2 task on its own line."""
    for task_id in discover_task_ids():
        print(task_id)


if __name__ == "__main__":
    main()
