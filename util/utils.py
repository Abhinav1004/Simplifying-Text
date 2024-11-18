# Import relevant libraries
import time
import json


def create_experiment_dir(repo_dir):
    """
    Create a unique experiment directory based on the current timestamp.

    Args:
        repo_dir: The repository directory location

    Returns:
        Path: The created experiment directory path
    """
    dir_name = '{}'.format(int(time.time() * 1000000))
    path = repo_dir / 'exp_{}'.format(dir_name)
    path.mkdir(parents=True, exist_ok=True)
    return path


def log_parameters(filepath, parameters):
    """
    Log parameters to a JSON file.

    Args:
        filepath: Path to the JSON file
        parameters: Parameters to log
    """
    with filepath.open('w') as f:
        json.dump({k: str(v) for k, v in parameters.items()}, f, indent=4)
