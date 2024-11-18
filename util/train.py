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


def train(model_config):
    """
    Function to train the model.

    Args:
        model_config: Dictionary containing model configurations.
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

    # Model initialization
    print("Initializing model...")
    model = Seq2SeqFineTunedModel(model_config)

    # Trainer setup and training
    trainer = pl.Trainer(**train_params)
    print("Starting training...")
    trainer.fit(model)
    print("Training finished.")

    # Saving the trained model
    output_dir = model_config['output_dir']
    model_save_path = os.path.join(output_dir, f"{model_name}-final")
    print(f"Saving model to {model_save_path}...")
    model.model.save_pretrained(model_save_path)
    print(f"Model saved at {model_save_path}.")

    return model.model, model.tokenizer, model_save_path
