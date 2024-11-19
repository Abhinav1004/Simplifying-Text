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


def save_log(output_dir, model_name, epoch, loss=None, sari=None, data_type='train'):
    """
    Save log for training or validation data.

    Args:
        output_dir: output directory
        model_name: use model name to save
        epoch (int): Current epoch.
        loss (float): Loss value (optional).
        sari (float): SARI score (optional).
        data_type: Data type train or validation
    """
    if data_type == 'train':
        with open('{}/{}_training_log.csv'.format(output_dir, model_name.replace("/", "-")), 'a') as f:
            log_line = f"{epoch},{loss if loss is not None else ''}\n"
            f.write(log_line)
    elif data_type == 'validation':
        with open('{}/{}_validation_log.csv'.format(output_dir, model_name.replace("/", "-")), 'a') as f:
            log_line = f"{epoch},{loss if loss is not None else ''},{sari if sari is not None else ''}\n"
            f.write(log_line)
