# Initialise user defined libraries
from util.model_runner import ModelRunner

if __name__ == "__main__":
    """
    Run the main functions to summarize text
    """
    configuration = {
        'seed': 0,
        'model_name': 'bart-baseline',
        'dataset_name': 'wiki_doc',
        'num_train_epochs': 10,
        'gpus': 1,
        'precision': 8,
        'gradient_clip_val': 1.0,
        'gradient_accumulation_steps': 1,
        'num_nodes': 1,
        'accelerator': 'mps',
        'dataset': 'wiki_doc'
    }

    # Initialize and run the trainer (example for BART model)
    model = ModelRunner(configuration)
    model.train_model()
    model.evaluate_model()
