from pathlib import Path
import sys
import time
import json

# Ensure the project root is added to the Python path
sys.path.append(str(Path(__file__).resolve().parent))

from preprocessor import WIKI_DOC, EXP_DIR
from Bart_baseline_finetuned import BartBaseLineFineTuned, train as bart_train
from T5_baseline_finetuned import T5BaseLineFineTuned, train as t5_train


class ModelTrainer:
    def __init__(
        self,
        model_name: str,
        seed: int = 42,
        max_epochs: int = 10,
        gpus: int = 1,
        precision: int = 8,
        gradient_clip_val: float = 1.0,
        accumulate_grad_batches: int = 0,
        num_nodes: int = 1,
        accelerator: str = 'mps',
    ):
        """
        Initialize the trainer with model configuration.

        Args:
            model_name (str): The name of the model ('bart' or 't5').
            seed (int): Randomization seed.
            max_epochs (int): Maximum number of training epochs.
            gpus (int): Number of GPUs to use.
            precision (int): Training precision (e.g., 16 for mixed precision).
            gradient_clip_val (float): Gradient clipping value.
            accumulate_grad_batches (int): Gradients are accumulated over N batches.
            num_nodes (int): Number of nodes for distributed training.
            accelerator (str): Type of accelerator (e.g., 'gpu', 'tpu', 'mps').
        """
        self.model_name = model_name.lower()
        self.model_class = None
        self.train_function = None

        if self.model_name == 'bart':
            self.model_class = BartBaseLineFineTuned
            self.train_function = bart_train
        elif self.model_name == 't5':
            self.model_class = T5BaseLineFineTuned
            self.train_function = t5_train
        else:
            raise ValueError("Invalid model name. Use 'bart' or 't5'.")

        # Store training parameters
        self.seed = seed
        self.max_epochs = max_epochs
        self.gpus = gpus
        self.precision = precision
        self.gradient_clip_val = gradient_clip_val
        self.accumulate_grad_batches = accumulate_grad_batches
        self.num_nodes = num_nodes
        self.accelerator = accelerator

    @staticmethod
    def create_experiment_dir() -> Path:
        """
        Create a unique experiment directory based on the current timestamp.

        Returns:
            Path: The created experiment directory path.
        """
        dir_name = f'{int(time.time() * 1000000)}'
        path = EXP_DIR / f'exp_{dir_name}'
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def log_parameters(filepath: Path, parameters: dict):
        """
        Log parameters to a JSON file.

        Args:
            filepath (Path): Path to the JSON file.
            parameters (dict): Parameters to log.
        """
        with filepath.open('w') as f:
            json.dump({k: str(v) for k, v in parameters.items()}, f, indent=4)

    def get_training_config(self):
        """
        Prepare the training configuration as a dictionary.

        Returns:
            dict: Training configuration.
        """
        return {
            'seed': self.seed,
            'max_epochs': self.max_epochs,
            'gpus': self.gpus,
            'precision': self.precision,
            'gradient_clip_val': self.gradient_clip_val,
            'accumulate_grad_batches': self.accumulate_grad_batches,
            'num_nodes': self.num_nodes,
            'accelerator': self.accelerator,
            'output_dir': self.create_experiment_dir(),
        }

    def train_model(self, dataset):
        """
        Run the training process.

        Args:
            dataset (str): The dataset to use for training.
        """
        # Prepare training arguments
        training_args = self.get_training_config()
        training_args['dataset'] = dataset

        # Log training arguments
        self.log_parameters(training_args['output_dir'] / "params.json", training_args)

        # Start training
        print(f"Starting training with {self.model_name.upper()} model on dataset: {dataset}")
        self.train_function(training_args)


if __name__ == "__main__":
    # Dataset configuration
    DATASET = WIKI_DOC

    # Initialize and run the trainer (example for BART model)
    trainer = ModelTrainer(
        model_name="bart",
        max_epochs=10
    )
    trainer.train_model(dataset=DATASET)
