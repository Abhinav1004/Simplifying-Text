# Import libraries
from pathlib import Path
import sys
import torch

# Import user-defined libraries
from util.utils import create_experiment_dir, log_parameters
from util.train import train
from util.evaluate_model.simsum_evaluator import SumSimEvaluator
from util.evaluate_model.evaluation_metrics import BartModelEvaluator, load_dataset
from util.simsum_models.simsum_model import SumSimModel
from util.baseline_models.baseline_model import Seq2SeqFineTunedModel
from util.generate_plots import plot_average_loss, plot_metric_distributions, identify_files


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

        # Store the model locations
        self.model_config['output_dir'] = create_experiment_dir(self.exp_dir, self.model_config)
        self.model_config['data_location'] = self.repo_dir / 'datasets'
        self.model_config['device'] = torch.device("cuda" if torch.cuda.is_available() else "mps")

        self.model_name = configuration['model_name'].lower()
        self.model_save_path = None
        self.model = None
        self.model_details = None
        self.select_model()

    def select_model(self):
        """
        Function to select the model class and configure the settings based on the model name.
        """
        if self.model_name == 'bart-baseline':
            self.model_config['model_name'] = 'facebook/bart-base' # 'Yale-LILY/brio-cnndm-uncased'
            self.model_config['scheduler_type'] = 'linear'
        elif self.model_name == 't5-baseline':
            self.model_config['model_name'] = 't5-base'
            self.model_config['scheduler_type'] = 'cosine'
        elif self.model_name == 'bart-simsum':
            self.model_config['summarizer_model_name'] = 'ainize/bart-base-cnn'
            self.model_config['simplifier_model_name'] = 'facebook/bart-base'
            self.model_config['scheduler_type'] = 'cosine'
            self.model = SumSimModel(
                self.model_config,
                summarizer_model_name=self.model_config['summarizer_model_name'],
                simplifier_model_name=self.model_config['simplifier_model_name']
            )
        elif self.model_name == 't5-simsum':
            self.model_config['summarizer_model_name'] = 't5-base'
            self.model_config['simplifier_model_name'] = 't5-base'
            self.model_config['scheduler_type'] = 'cosine'
            self.model = SumSimModel(
                self.model_config,
                summarizer_model_name=self.model_config['summarizer_model_name'],
                simplifier_model_name=self.model_config['simplifier_model_name']
            )
        else:
            raise ValueError("Invalid model name. Use 'bart-baseline', 't5-baseline', 'bart-simsum', or 't5-simsum'.")

    def train_model(self):
        """
        Run the training process.
        """
        # Log training arguments
        log_parameters(self.model_config['output_dir'] / "params.json", self.model_config)

        # Start training
        print(
            f"Starting training with {self.model_name.upper()} model on dataset: {self.model_config['dataset']}"
        )
        if self.model is None:
            # Initialize model if not already set (for baseline models)
            self.model = Seq2SeqFineTunedModel(self.model_config)
        self.model, self.model_save_path = train(self.model_config, self.model)

    def evaluate_model(self):
        """
        Function to evaluate the model.
        """
        print("Starting evaluation of models")
        if self.model_name in ['bart-simsum', 't5-simsum']:
            evaluator = SumSimEvaluator(self.model_config, self.model.summarizer, self.model.simplifier,
                                        self.model.summarizer_tokenizer, self.model.simplifier_tokenizer)
        else:
            # Use standard evaluator for baseline models
            evaluator = BartModelEvaluator(self.model_config, self.model.model, self.model.tokenizer)

        # Load datasets (D_Wiki and Wiki_Doc)
        dataset_dir = self.model_config['data_location']
        dataset_name = self.model_config['dataset']

        print(f"Evaluating on {dataset_name}")
        complex_sents, simple_sents = load_dataset(
            dataset_dir, dataset_name, percentage=self.model_config['test_sample_size']
        )
        scores, score_table = evaluator.evaluate(complex_sents, simple_sents)
        print(f"Results for {dataset_name}: {scores}")

        # Generate plots
        files = identify_files(self.model_config['output_dir'])
        plot_average_loss(self.model_config['output_dir'], files['training_log'], files['validation_log'])
        plot_metric_distributions(self.model_config['output_dir'], files['evaluation_metrics'])

