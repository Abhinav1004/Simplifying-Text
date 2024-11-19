# Initialise user defined libraries
from model_runner import ModelRunner

if __name__ == "__main__":
    """
    Run the main functions to summarize text
    """
    configuration = {
        'seed': 0,
        'gradient_accumulation_steps': 1,
        'learning_rate': 1e-5,
        'max_seq_length': 256,
        'adam_epsilon': 1e-8,
        'weight_decay': 0.0001,
        'warmup_steps': 5,
        'custom_loss': True,
        'train_sample_size': 0.1,
        'valid_sample_size': 0.1,
        'hidden_size': 1,
        'w1': 1,

        # To edit
        'model_name': 'bart-simsum',
        'dataset': 'wiki_doc',
        'num_train_epochs': 10,
        'train_batch_size': 8,
        'valid_batch_size': 8,
        'lambda_': 0.001,
        'prompting_strategy': 'no_prompting',
        'div_score': 0.5,
        'top_keywords': 5,
        'test_sample_size': 0.2
    }

    # Initialize, run and evaluate the model
    model = ModelRunner(configuration)
    model.train_model()
    model.evaluate_model()
