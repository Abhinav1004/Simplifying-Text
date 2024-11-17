import os
import logging
import torch
from time import time
from pytorch_lightning.loggers import TensorBoardLogger
from easse.sari import corpus_sari
from preprocessor import yield_lines, read_lines
from preprocessor import  get_data_filepath
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.trainer import seed_everything
from transformers import (
    AdamW,
    BartForConditionalGeneration, BartTokenizer,
    get_linear_schedule_with_warmup,
)

class BartBaseLineFineTuned(pl.LightningModule):
    """
    A PyTorch Lightning module for fine-tuning a BART model for sequence-to-sequence tasks like summarization.

    Args:
        model_name (str): Pre-trained BART model to fine-tune (e.g., 'facebook/bart-large-cnn').
        train_batch_size (int): Batch size for training.
        valid_batch_size (int): Batch size for validation.
        learning_rate (float): Learning rate for the optimizer.
        max_seq_length (int): Maximum sequence length for input text.
        adam_epsilon (float): Epsilon value for Adam optimizer.
        weight_decay (float): Weight decay for optimizer.
        warmup_steps (int): Number of warmup steps for learning rate scheduler.
        num_train_epochs (int): Number of training epochs.
        custom_loss (bool): Whether to use a custom loss function.
        gradient_accumulation_steps (int): Number of gradient accumulation steps.
        train_sample_size (float): Sample size for training dataset.
        valid_sample_size (float): Sample size for validation dataset.
        device (str): The device to run the model on ('cpu', 'cuda', 'mps').
        dataset (Dataset): Dataset for training and validation.
    """

    def __init__(self,
                 training_parameters,
                 model_name='Yale-LILY/brio-cnndm-uncased',
                 train_batch_size=4,
                 valid_batch_size=4,
                 learning_rate=1e-5,
                 max_seq_length=256,
                 adam_epsilon=1e-8,
                 weight_decay=0.0001,
                 warmup_steps=5,
                 num_train_epochs=10,
                 custom_loss=False,
                 gradient_accumulation_steps=1,
                 train_sample_size=0.01,
                 valid_sample_size=0.01,
                 device='mps', dataset=None):
        super(BartBaseLineFineTuned, self).__init__()

        # Store hyperparameters
        self.save_hyperparameters()

        # Initialize the BART model and tokenizer
        self.training_parameters = training_parameters
        print("Training Parameters ",self.training_parameters)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.model = self.model.to(device)
        self.tokenizer = BartTokenizer.from_pretrained(model_name)

        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.learning_rate = learning_rate
        self.max_seq_length = max_seq_length
        self.adam_epsilon = adam_epsilon
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.num_train_epochs = num_train_epochs
        self.custom_loss = custom_loss
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.train_sample_size = train_sample_size
        self.valid_sample_size = valid_sample_size
        self.device_name = device

        self.dataset = self.training_parameters['dataset']

        self.model_name = str(time())
        self.core_model_path = self.model_name + '_core'
        self.output_dir = self.training_parameters['output_dir']
        self.model_store_path = self.output_dir / self.core_model_path

    def is_logger(self):
        """
        Returns True if this is the first rank (for distributed training), False otherwise.
        """
        return self.trainer.global_rank <= 0

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None,
                decoder_attention_mask=None, labels=None):
        """
        Defines the forward pass of the model.

        Args:
            input_ids (tensor): Input tensor containing tokenized input IDs.
            attention_mask (tensor): Attention mask for the input.
            decoder_input_ids (tensor): Decoder input IDs for sequence generation.
            decoder_attention_mask (tensor): Attention mask for the decoder.
            labels (tensor): Target labels for training.

        Returns:
            ModelOutput: Model's output, including loss if labels are provided.
        """
        return self.model(input_ids=input_ids,
                          attention_mask=attention_mask,
                          decoder_input_ids=decoder_input_ids,
                          decoder_attention_mask=decoder_attention_mask,
                          labels=labels)

    def training_step(self, batch, batch_idx):
        """
        Performs a training step, computes loss, and logs the results.

        Args:
            batch (dict): Batch of training data.
            batch_idx (int): Index of the current batch.

        Returns:
            Tensor: Loss value for the current batch.
        """
        source = batch["source"]
        labels = batch['target_ids']

        # Ignore padding tokens in loss calculation
        labels[labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(input_ids=batch["source_ids"],
                       attention_mask=batch["source_mask"],
                       labels=labels,
                       decoder_attention_mask=batch["target_mask"])

        loss = outputs.loss
        self.log('train_loss', loss, on_step=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Performs a validation step, computes loss, and logs the results.

        Args:
            batch (dict): Batch of validation data.
            batch_idx (int): Index of the current batch.

        Returns:
            Tensor: Loss value for the current batch.
        """
        loss = self.sari_validation_step(batch)
        self.log('val_loss', loss, batch_size=self.valid_batch_size)
        return loss

    def sari_validation_step(self, batch):
        """
        Calculates the SARI score (Summarization Accuracy with Respect to ROUGE) for the validation batch.

        Args:
            batch (dict): Batch of validation data.

        Returns:
            float: SARI score for the batch.
        """

        def generate(sentence):
            encoding = self.tokenizer(
                [sentence],
                max_length=self.max_seq_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            ).to(self.device)

            input_ids = encoding['input_ids']
            attention_mask = encoding['attention_mask']

            beam_outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                max_length=256,
                num_beams=5,
                top_k=120,
                top_p=0.95,
                early_stopping=True,
                num_return_sequences=1
            ).to(self.device)

            return self.tokenizer.decode(beam_outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

        pred_sents = [generate(source) for source in batch["source"]]
        score = corpus_sari(batch["source"], pred_sents, [batch["targets"]])

        print("\nSARI score: ", score)
        return 1 - score / 100

    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler.

        Returns:
            list: A list containing the optimizer and scheduler.
        """
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate, eps=self.adam_epsilon)

        # Calculate the total training steps
        t_total = (
                (
                        len(self.train_dataloader().dataset) // self.train_batch_size
                ) // self.gradient_accumulation_steps
                * float(self.num_train_epochs)
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=t_total
        )

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step', 'frequency': 1}]

    def save_core_model(self):
        """
        Saves the fine-tuned model and tokenizer to the specified directory.
        """
        self.model.save_pretrained(self.model_store_path)
        self.tokenizer.save_pretrained(self.model_store_path)

    def train_dataloader(self):
        """
        Returns the training DataLoader.

        Returns:
            DataLoader: The training DataLoader.
        """
        train_dataset = TrainDataset(
            dataset=self.dataset,
            tokenizer=self.tokenizer,
            max_len=self.max_seq_length,
            sample_size=self.train_sample_size,
        )
        dataloader = DataLoader(
            train_dataset,
            batch_size=self.train_batch_size,
            drop_last=True,
            shuffle=True,
            pin_memory=True,
            num_workers=0
        )
        return dataloader

    def val_dataloader(self):
        """
        Returns the validation DataLoader.

        Returns:
            DataLoader: The validation DataLoader.
        """
        val_dataset = ValDataset(
            dataset=self.dataset,
            tokenizer=self.tokenizer,
            max_len=self.max_seq_length,
            sample_size=self.valid_sample_size
        )
        return DataLoader(
            val_dataset,
            batch_size=self.valid_batch_size
        )


logger = logging.getLogger(__name__)


class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        logger.info("***** Validation results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
            # Log results
            for key in sorted(metrics):
                print(key, metrics[key])
                if key not in ["log", "progress_bar"]:
                    logger.info("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer, pl_module):
        logger.info("***** Test results *****")

        if pl_module.is_logger():
            metrics = trainer.callback_metrics

            # Log and save results to file
            output_test_results_file = os.path.join(pl_module.args.output_dir, "test_results.txt")
            with open(output_test_results_file, "w") as writer:
                for key in sorted(metrics):
                    if key not in ["log", "progress_bar"]:
                        logger.info("{} = {}\n".format(key, str(metrics[key])))
                        writer.write("{} = {}\n".format(key, str(metrics[key])))


##### build dataset Loader #####
class TrainDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_len=256, sample_size=1):
        self.sample_size = sample_size
        self.source_filepath = get_data_filepath(dataset,'train','complex')
        self.target_filepath = get_data_filepath(dataset,'train','simple')

        self.max_len = max_len
        self.tokenizer = tokenizer

        self._load_data()

    def _load_data(self):
        self.inputs = read_lines(self.source_filepath)
        self.targets = read_lines(self.target_filepath)

    def __len__(self):
        return int(len(self.inputs) * self.sample_size)

    def __getitem__(self, index):
        source = self.inputs[index]
        target = self.targets[index]

        tokenized_inputs = self.tokenizer(
            [source],
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_tensors="pt"
        )
        tokenized_targets = self.tokenizer(
            [target],
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_tensors="pt"
        )
        source_ids = tokenized_inputs["input_ids"].squeeze()
        target_ids = tokenized_targets["input_ids"].squeeze()

        src_mask = tokenized_inputs["attention_mask"].squeeze()  # might need to squeeze
        target_mask = tokenized_targets["attention_mask"].squeeze()  # might need to squeeze

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask,
                'sources': self.inputs[index], 'targets': [self.targets[index]],
                'source': source, 'target': target}


class ValDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_len=256, sample_size=1):
        self.sample_size = sample_size
        ### WIKI-large dataset ###
        self.source_filepath = get_data_filepath(dataset, 'valid', 'complex')
        self.target_filepaths = get_data_filepath(dataset, 'valid', 'simple')
        print(self.source_filepath)

        self.max_len = max_len
        self.tokenizer = tokenizer

        self._build()

    def __len__(self):
        return int(len(self.inputs) * self.sample_size)

    def __getitem__(self, index):
        return {"source": self.inputs[index], "targets": self.targets[index]}

    def _build(self):
        self.inputs = []
        self.targets = []

        for source in yield_lines(self.source_filepath):
            self.inputs.append(source)
        
        for target in yield_lines(self.target_filepaths):
            self.targets.append(target)



def train(args):
    seed_everything(args['seed'])

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args['output_dir'],
        filename="checkpoint-{epoch}",
        monitor="val_loss",
        verbose=True,
        mode="min",
        save_top_k=1
    )
    bar_callback = pl.callbacks.TQDMProgressBar(refresh_rate=1)
    train_params = dict(
        accumulate_grad_batches=args['gradient_accumulation_steps'],
        max_epochs=args['num_train_epochs'],
        callbacks=[
            LoggingCallback(),
            checkpoint_callback, bar_callback],
        logger=TensorBoardLogger(f"{args['output_dir']}/logs"),
        num_sanity_val_steps=0,  # skip sanity check to save time for debugging purpose
    )

    print("Initialize model")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    print(f"Device used {device}")
    model = BartBaseLineFineTuned(args)
 
    #model = T5FineTuner(**train_args)
    trainer = pl.Trainer(**train_params)

    print(" Training model")
    trainer.fit(model)

    print("training finished")

    print("Saving model")
    model.model.save_pretrained(args['output_dir'])
