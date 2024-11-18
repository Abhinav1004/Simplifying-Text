# Import libraries
from pathlib import Path
import sys
import torch

# Import user defined libraries
from util.utils import create_experiment_dir, log_parameters
from util.train import train
from util.evaluate_model.evaluation_metrics import BartModelEvaluator, load_dataset


class ModelRunner:
    def __init__(self, configuration):
        """
        Initialize the trainer with model configuration.

        Args:
            configuration: The dictionary containing model configurations
        """
        # Ensure the project root is added to the Python path
        sys.path.append(str(Path(__file__).resolve().parent))

        # Initialise the output directory
        self.repo_dir = Path(__file__).resolve().parent
        self.exp_dir = self.repo_dir / 'outputs'

        # Define the model name
        self.model_config = configuration.copy()
        self.model_name = configuration['model_name'].lower()
        self.model_save_path = None
        self.model = None
        self.tokenizer = None
        self.model_details = None
        self.select_model()

        # Store the model locations
        self.model_config['output_dir'] = create_experiment_dir(self.exp_dir)
        self.model_config['data_location'] = self.repo_dir / 'datasets'
        self.model_config['device'] = torch.device("cuda" if torch.cuda.is_available() else "mps")

    def select_model(self):
        """
        Function to select the model class and train function given the model name
        """
        if self.model_name == 'bart-baseline':
            self.model_config['model_name'] = 'Yale-LILY/brio-cnndm-uncased'
            self.model_config['scheduler_type'] = 'linear'
        elif self.model_name == 't5-baseline':
            self.model_config['model_name'] = 't5-base'
            self.model_config['scheduler_type'] = 'cosine'
        # elif self.model_name == 'bart-simsum':
        #     self.model_class = BartSimSum
        # elif self.model_name == 't5-simsum':
        #     self.model_class = T5SimSum
        else:
            raise ValueError("Invalid model name. Use 'bart-baseline', 't5-baseline', 'bart-simsum' or 't5-simsum'.")

    def train_model(self):
        """
        Run the training process.
        """
        # Log training arguments
        log_parameters(self.model_config['output_dir'] / "params.json", self.model_config)

        # Start training
        print(
            "Starting training with {} model on dataset: {}".format(
                self.model_config['model_name'].upper(),
                self.model_config['dataset']
            )
        )
        self.model, self.tokenizer, self.model_save_path = train(self.model_config)

    def evaluate_model(self):
        """
        Function to evaluate the model
        """
        print("Starting evaluation of models")
        # Initialize the evaluator
        evaluator = BartModelEvaluator(self.model_config, self.model, self.tokenizer)

        # Load datasets (D_Wiki and Wiki_Doc)
        dataset_dir = self.model_config['data_location']
        dataset_name = self.model_config['dataset']

        print(f"Evaluating on {dataset_name}")
        complex_sents, simple_sents = load_dataset(dataset_dir, dataset_name)
        scores = evaluator.evaluate(complex_sents, simple_sents)
        print(f"Results for {dataset_name}: {scores}")

