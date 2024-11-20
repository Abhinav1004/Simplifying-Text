# Import relevant libraries
import time
import json


def create_experiment_dir(repo_dir, model_config):
    """
    Create a unique experiment directory based on the current timestamp and model configuration.

    Args:
        repo_dir: The repository directory location
        model_config: The configuration dictionary to include in the directory name

    Returns:
        Path: The created experiment directory path
    """
    # Find the next folder number based on existing directories
    existing_dirs = [d for d in repo_dir.iterdir() if d.is_dir() and d.name.startswith('exp_')]
    next_number = 1 + max(
        (int(d.name.split('_')[0]) for d in existing_dirs if d.name.split('_')[0].isdigit()),
        default=0
    )

    # Construct a directory name with model configuration values
    dir_name = (
        f"{next_number}_"
        f"{model_config['model_name']}_"
        f"{model_config['dataset']}_"
        f"epochs-{model_config['num_train_epochs']}_"
        f"batch-{model_config['train_batch_size']}_"
        f"val_batch-{model_config['valid_batch_size']}_"
        f"lambda-{model_config['lambda_']}_"
        f"prompt-{model_config['prompting_strategy']}_"
        f"div-{model_config['div_score']}_"
        f"keywords-{model_config['top_keywords']}_"
        f"test_sample-{model_config['test_sample_size']}"
    )
    path = repo_dir / dir_name
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
