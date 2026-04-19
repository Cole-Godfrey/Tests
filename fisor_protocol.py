FISOR_PAPER_EVAL_EPISODES = 20


def get_fisor_paper_cost_limit(task: str) -> int:
    """
    Return the published FISOR evaluation cost limit for a DSRL task.

    FISOR's paper states:
    - Safety-Gymnasium tasks use cost limit 10
    - Other DSRL environments use cost limit 5
    """
    if "Gymnasium" in task:
        return 10
    if task.startswith("OfflineMetadrive-"):
        return 5
    if task.startswith("Offline"):
        return 5
    raise ValueError(f"Unsupported task for FISOR paper protocol: {task}")
