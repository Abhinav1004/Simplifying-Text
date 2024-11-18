# Import libraries
from pathlib import Path
import sys
import torch

# Import user defined libraries
from util.utils import create_experiment_dir, log_parameters
from util.train import train
from util.baseline_models.bart_baseline import BartBaseLineFineTuned
# from util.simsum_models.bart_simsum import SumSim as BartSimSum
# from util.baseline_models.t5_baseline import T5BaseLineFineTuned
# from util.simsum_models.t5_simsum import SumSim as T5SimSum


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
        self.model_name = configuration['model_name'].lower()
        self.model_class = None
        self.select_model()

        # Store the model configuration
        self.model_config = configuration.copy()
        self.model_config['output_dir'] = create_experiment_dir(self.exp_dir)
        self.model_config['data_location'] = self.repo_dir / 'datasets'
        self.model_config['device'] = torch.device("cuda" if torch.cuda.is_available() else "mps")

    def select_model(self):
        """
        Function to select the model class and train function given the model name
        """
        if self.model_name == 'bart-baseline':
            self.model_class = BartBaseLineFineTuned
        # elif self.model_name == 't5-baseline':
        #     self.model_class = T5BaseLineFineTuned
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
        train(self.model_config, self.model_class)

    @staticmethod
    def evaluate_model():
        """
        Function to evaluate the model
        """
        print("Evaluation not complete")

