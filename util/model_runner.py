# Import libraries
from pathlib import Path
import sys

# Import user defined libraries
from util.utils import create_experiment_dir, log_parameters
from util.baseline_models.bart_baseline import BartBaseLineFineTuned, train as bart_train
from util.simsum_models.bart_simsum import SumSim as BartSimSum, train as bart_simsum_train
from util.baseline_models.t5_baseline import T5BaseLineFineTuned, train as t5_train
from util.simsum_models.t5_simsum import SumSim as T5SimSum, train as t5_simsum_train


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
        self.exp_dir = self.repo_dir / 'experiments'

        # Define the model name
        self.model_name = configuration['model_name'].lower()
        self.model_class = None
        self.train_function = None
        self.select_model()

        # Store the model configuration
        self.model_config = configuration.copy()
        self.model_config['output_dir'] = create_experiment_dir(self.repo_dir)

    def select_model(self):
        """
        Function to select the model class and train function given the model name
        """
        if self.model_name == 'bart-baseline':
            self.model_class = BartBaseLineFineTuned
            self.train_function = bart_train
        elif self.model_name == 't5-baseline':
            self.model_class = T5BaseLineFineTuned
            self.train_function = t5_train
        elif self.model_name == 'bart-simsum':
            self.model_class = BartSimSum
            self.train_function = bart_simsum_train
        elif self.model_name == 't5-simsum':
            self.model_class = T5SimSum
            self.train_function = t5_simsum_train
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
        self.train_function(self.model_config)

    def evaluate_model(self):
        """
        Function to evaluate the model
        """

