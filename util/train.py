# Import relevant libraries
import os
import logging
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from util.baseline_models.baseline_model import Seq2SeqFineTunedModel

logger = logging.getLogger(__name__)


class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        """
        Logs validation results at the end of each validation epoch.
        """
        logger.info("***** Validation results *****")
        if hasattr(pl_module, "is_logger") and pl_module.is_logger():
            metrics = trainer.callback_metrics
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    logger.info(f"{key} = {metrics[key]}\n")
                    print(f"{key}: {metrics[key]}")

    def on_test_end(self, trainer, pl_module):
        """
        Logs and saves test results to a file at the end of testing.
        """
        logger.info("***** Test results *****")
        if hasattr(pl_module, "is_logger") and pl_module.is_logger():
            metrics = trainer.callback_metrics
            output_file = os.path.join(pl_module.args.output_dir, "test_results.txt")
            with open(output_file, "w") as writer:
                for key in sorted(metrics):
                    if key not in ["log", "progress_bar"]:
                        logger.info(f"{key} = {metrics[key]}\n")
                        writer.write(f"{key} = {metrics[key]}\n")


def train(model_config, model_instance=None):
    """
    Function to train the model.

    Args:
        model_config: Dictionary containing model configurations.
        model_instance: Instance of the model to be trained (optional).
    """
    # Seed for reproducibility
    seed = model_config.get('seed', 42)
    pl.seed_everything(seed)

    # Model checkpointing configuration
    model_name = model_config.get('model_name')
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=model_config['output_dir'],
        filename=f"{model_name}-checkpoint-{{epoch}}",
        monitor="val_loss",
        verbose=True,
        mode="min",
        save_top_k=1
    )
    # Progress bar callback
    bar_callback = pl.callbacks.TQDMProgressBar(refresh_rate=1)

    # Training parameters
    train_params = {
        'accumulate_grad_batches': model_config.get('gradient_accumulation_steps', 1),
        'max_epochs': model_config.get('num_train_epochs', 5),
        'callbacks': [LoggingCallback(), checkpoint_callback, bar_callback],
        'logger': TensorBoardLogger(f"{model_config['output_dir']}/logs"),
        'num_sanity_val_steps': 0
    }

    # Model initialization (if model instance is not provided)
    if model_instance is None:
        print("Initializing baseline model...")
        model = Seq2SeqFineTunedModel(model_config)
    else:
        model = model_instance

    # Trainer setup and training
    trainer = pl.Trainer(**train_params)
    print("Starting training...")
    trainer.fit(model)
    print("Training finished.")

    # Saving the trained model
    output_dir = model_config['output_dir']
    if "simsum" in model_name:
        summarizer_save_path = os.path.join(output_dir, f"{model_name}-summarizer-final")
        simplifier_save_path = os.path.join(output_dir, f"{model_name}-simplifier-final")
        print(f"Saving summarizer to {summarizer_save_path}...")
        model.summarizer.save_pretrained(summarizer_save_path)
        model.summarizer_tokenizer.save_pretrained(summarizer_save_path)
        print(f"Summarizer saved at {summarizer_save_path}.")

        print(f"Saving simplifier to {simplifier_save_path}...")
        model.simplifier.save_pretrained(simplifier_save_path)
        model.simplifier_tokenizer.save_pretrained(simplifier_save_path)
        print(f"Simplifier saved at {simplifier_save_path}.")
    else:
        model_save_path = os.path.join(output_dir, f"{model_name}-final")
        print(f"Saving model to {model_save_path}...")
        model.model.save_pretrained(model_save_path)
        model.tokenizer.save_pretrained(model_save_path)
        print(f"Model saved at {model_save_path}.")

    return model, output_dir
