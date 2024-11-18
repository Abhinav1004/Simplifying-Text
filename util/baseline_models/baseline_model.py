# Import necessary libraries
from torch.utils.data import DataLoader
from util.train_valid_data_generation import TrainDataset, ValDataset
import pytorch_lightning as pl
from transformers import (
    AdamW,
    AutoModelForSeq2SeqLM, AutoTokenizer,
    get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
)
from easse.sari import corpus_sari


class Seq2SeqFineTunedModel(pl.LightningModule):
    """
    A generic PyTorch Lightning module for fine-tuning a sequence-to-sequence model for summarization or text simplification.

    Args:
        training_parameters (dict): Dictionary of training parameters.
        model_name (str): Pre-trained model to fine-tune (e.g., 't5-base', 'Yale-LILY/brio-cnndm-uncased').
    """
    def __init__(self, training_parameters, model_name='t5-base'):
        super(Seq2SeqFineTunedModel, self).__init__()

        # Store hyperparameters and initialize model and tokenizer
        self.save_hyperparameters()
        self.training_parameters = training_parameters
        self.device_name = training_parameters['device']

        # Initialize parameters from the training dictionary
        self.model_name = training_parameters['model_name']
        self.train_batch_size = training_parameters['train_batch_size']
        self.valid_batch_size = training_parameters['valid_batch_size']
        self.learning_rate = training_parameters['learning_rate']
        self.max_seq_length = training_parameters['max_seq_length']
        self.adam_epsilon = training_parameters['adam_epsilon']
        self.weight_decay = training_parameters['weight_decay']
        self.warmup_steps = training_parameters['warmup_steps']
        self.train_sample_size = training_parameters['train_sample_size']
        self.valid_sample_size = training_parameters['valid_sample_size']
        self.num_train_epochs = training_parameters['num_train_epochs']
        self.gradient_accumulation_steps = training_parameters['gradient_accumulation_steps']
        self.custom_loss = training_parameters.get('custom_loss', False)
        self.scheduler_type = training_parameters.get('scheduler_type', 'linear')

        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Data and output paths
        self.dataset = self.training_parameters['dataset']
        self.data_location = self.training_parameters['data_location']
        self.model_store_path = training_parameters['output_dir'] / (model_name + '_fine_tuned')

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

        if self.scheduler_type == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=t_total
            )
        else:
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=t_total
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
            data_set_dir=self.data_location,
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
            data_set_dir=self.data_location,
            dataset=self.dataset,
            tokenizer=self.tokenizer,
            max_len=self.max_seq_length,
            sample_size=self.valid_sample_size
        )
        return DataLoader(
            val_dataset,
            batch_size=self.valid_batch_size
        )
