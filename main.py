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
        'custom_loss': True, # [True,False]
        'train_sample_size': 0.1,
        'valid_sample_size': 0.1,
        'hidden_size': 1,
        'w1': 1,

        # Edit the following parameters as per the requirement i.e dataset,model type,epochs,batch size etc.
        'model_name': 'bart-simsum', # ['bart-baseline', 't5-baseline','bart-simsum', 't5-simsum']
        'dataset': 'wiki_doc', # ['wiki_doc', 'd_wiki']
        'num_train_epochs': 10,
        'train_batch_size': 8,
        'valid_batch_size': 8,
        'lambda_': 0.001, # for simsum loss function
        'prompting_strategy': 'no_prompting', # ['no_prompting', 'kw_score']
        'div_score': 0.5, # for 'kw_score' prompting_strategy
        'top_keywords': 5, # for 'kw_score' prompting_strategy
        'test_sample_size': 0.2
    }

    # Initialize, run and evaluate the model
    model = ModelRunner(configuration)
    model.train_model()
    model.evaluate_model()
