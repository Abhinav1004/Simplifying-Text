# Import necessary libraries
# Import necessary libraries
from torch.utils.data import DataLoader
from util.simsum_models.keyword_prompting import create_kw_sep_prompt, create_kw_score_prompt
from util.train_valid_data_generation import TrainDataset, ValDataset
import pytorch_lightning as pl
from transformers import (
    AdamW, AutoModelForSeq2SeqLM, AutoTokenizer,
    get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
)
from easse.sari import corpus_sari
import torch
import torch.nn as nn
from util.utils import save_log


class SumSimModel(pl.LightningModule):
    """
    A PyTorch Lightning model for fine-tuning summarization and simplification using a sequence-to-sequence model
    with keyword prompting and custom loss.

    Args:
        training_parameters (dict): Dictionary of training parameters.
        summarizer_model_name (str): Pre-trained summarization model.
        simplifier_model_name (str): Pre-trained simplification model.
    """
    def __init__(self, training_parameters, summarizer_model_name='t5-base', simplifier_model_name='t5-base'):
        super(SumSimModel, self).__init__()

        # Store hyperparameters and initialize model and tokenizer
        self.save_hyperparameters()
        self.training_parameters = training_parameters
        self.device_name = training_parameters['device']

        # Initialize parameters from the training dictionary
        self.summarizer_model_name = training_parameters.get('summarizer_model_name', summarizer_model_name)
        self.simplifier_model_name = training_parameters.get('simplifier_model_name', simplifier_model_name)
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
        self.lambda_ = training_parameters.get('lambda_', 1)
        self.hidden_size = training_parameters.get('hidden_size', 1)
        self.w1 = training_parameters.get('w1', 1)
        self.top_keywords = training_parameters['top_keywords']
        self.div_score = training_parameters['div_score']
        self.prompting_strategy = training_parameters.get('prompting_strategy', 'kw_score')
        with open('{}/{}_training_log.csv'.format(
                self.training_parameters['output_dir'],
                self.training_parameters['model_name']
        ), 'w') as f: f.write('epoch,loss\n')
        with open('{}/{}_validation_log.csv'.format(
                self.training_parameters['output_dir'],
                self.training_parameters['model_name']
        ), 'w') as f: f.write('epoch,loss,sari\n')

        # Initialize summarizer and simplifier
        self.summarizer = AutoModelForSeq2SeqLM.from_pretrained(self.summarizer_model_name).to(self.device_name)
        self.summarizer_tokenizer = AutoTokenizer.from_pretrained(self.summarizer_model_name)

        self.simplifier = AutoModelForSeq2SeqLM.from_pretrained(self.simplifier_model_name).to(self.device_name)
        self.simplifier_tokenizer = AutoTokenizer.from_pretrained(self.simplifier_model_name)

        # Custom weight matrix for embedding similarity
        self.W = torch.randn((768, int(self.hidden_size)), requires_grad=True, device=self.device_name)
        self.CosSim = nn.CosineSimilarity(dim=2, eps=1e-6)
        self.relu = nn.ReLU()

        # Data and output paths
        self.dataset = self.training_parameters['dataset']
        self.data_location = self.training_parameters['data_location']
        self.model_store_path = training_parameters['output_dir'] / (training_parameters['model_name'] + '_fine_tuned')

    def is_logger(self):
        """
        Returns True if this is the first rank (for distributed training), False otherwise.
        """
        return self.trainer.global_rank <= 0

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, labels=None):
        """
        Forward pass of the simplifier model.
        """
        return self.simplifier(input_ids=input_ids,
                               attention_mask=attention_mask,
                               decoder_input_ids=decoder_input_ids,
                               decoder_attention_mask=decoder_attention_mask,
                               labels=labels)

    def training_step(self, batch, batch_idx):
        """
        Performs a training step, computes loss based on summarizer and simplifier stages, and logs the results.

        Args:
            batch (dict): Batch of training data.
            batch_idx (int): Index of the current batch.

        Returns:
            Tensor: Loss value for the current batch.
        """
        source = batch["source"]
        labels = batch['target_ids']
        targets = batch['target']
        labels[labels[:, :] == self.simplifier_tokenizer.pad_token_id] = -100

        # Select the keyword prompting strategy based on training parameters
        if self.training_parameters.get('prompting_strategy') == 'kw_score':
            prompt_source = [create_kw_score_prompt(text, self.top_keywords, self.div_score) for text in source]
        else:
            prompt_source = [create_kw_sep_prompt(text, self.top_keywords, self.div_score) for text in source]

        # Tokenize targets for the simplifier
        targets_encoding = self.simplifier_tokenizer(
            targets,
            max_length=256,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        tgt_ids = targets_encoding['input_ids'].to(self.device_name)
        tgt_mask = targets_encoding['attention_mask'].to(self.device_name)

        # Forward pass through the simplifier
        tgt_output = self.simplifier(
            input_ids=tgt_ids,
            attention_mask=tgt_mask,
            labels=labels,
            decoder_attention_mask=batch['target_mask']
        )
        H_sim = tgt_output.encoder_last_hidden_state

        # Summarizer stage
        inputs = self.summarizer_tokenizer(
            prompt_source,
            max_length=512,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        src_ids = inputs['input_ids'].to(self.device_name)
        src_mask = inputs['attention_mask'].to(self.device_name)

        # Forward pass through the summarizer
        sum_outputs = self.summarizer(
            input_ids=src_ids,
            attention_mask=src_mask,
            labels=labels,
            decoder_attention_mask=batch['target_mask']
        )

        # Generate summary
        summary_ids = self.summarizer.generate(
            inputs['input_ids'].to(self.device_name),
            do_sample=True,
            num_beams=5,
            min_length=10,
            max_length=256
        ).to(self.device_name)

        # Pad summaries for simplifier input
        padded_summary_ids = torch.zeros((summary_ids.shape[0], 256), dtype=torch.long).fill_(
            self.simplifier_tokenizer.pad_token_id).to(self.device_name)
        for i, summary_id in enumerate(summary_ids):
            padded_summary_ids[i, :summary_id.shape[0]] = summary_id

        summary_attention_mask = torch.ones(padded_summary_ids.shape).to(self.device_name)
        summary_attention_mask[padded_summary_ids[:, :] == self.simplifier_tokenizer.pad_token_id] = 0

        # Forward pass through the simplifier with summaries
        sim_outputs = self(
            input_ids=padded_summary_ids,
            attention_mask=summary_attention_mask,
            labels=labels,
            decoder_attention_mask=batch['target_mask']
        )
        H2 = sim_outputs.encoder_last_hidden_state

        # Compute similarity
        Rep1 = torch.matmul(H_sim, self.W)
        Rep2 = torch.matmul(H2, self.W)
        Rep1 = self.relu(Rep1)
        Rep2 = self.relu(Rep2)
        sim_score = self.CosSim(Rep1, Rep2)

        # Custom loss logic
        if self.training_parameters.get('custom_loss'):
            loss = sim_outputs.loss * self.training_parameters['w1']
            loss += (-self.training_parameters['lambda_'] * (sim_score.mean(dim=1).mean(dim=0)))
            self.log('train_loss', sim_outputs.loss, on_step=True, prog_bar=True, logger=True)
        else:
            loss = sim_outputs.loss
            self.log('train_loss', loss, on_step=True, prog_bar=True, logger=True)

        # Save loss for each data point
        save_log(
            self.training_parameters['output_dir'],
            self.training_parameters['model_name'],
            self.current_epoch,
            loss=loss.item(),
            data_type='train'
        )

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Performs a validation step, computes and accumulates SARI scores, and logs the results.

        Args:
            batch (dict): Batch of validation data.
            batch_idx (int): Index of the current batch.

        Returns:
            float: SARI score for the current batch.
        """
        loss = self.sari_validation_step(batch)
        self.log('val_loss', loss, batch_size=self.training_parameters['valid_batch_size'])

        return torch.tensor(loss, dtype=float)

    def sari_validation_step(self, batch):
        """
        Calculates the SARI score (Summarization Accuracy with Respect to ROUGE) for the validation batch.

        Args:
            batch (dict): Batch of validation data.

        Returns:
            float: SARI score for the batch.
        """
        def generate(sentence):
            inputs = self.summarizer_tokenizer(
                ["summarize: " + sentence],
                max_length=512,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            summary_ids = self.summarizer.generate(
                inputs['input_ids'].to(self.device_name),
                num_beams=15,
                max_length=256,
                top_k=130,
                top_p=0.95
            ).to(self.device_name)

            summary_attention_mask = torch.ones(summary_ids.shape).to(self.device_name)
            summary_attention_mask[summary_ids[:, :] == self.summarizer_tokenizer.pad_token_id] = 0

            beam_outputs = self.simplifier.generate(
                input_ids=summary_ids,
                attention_mask=summary_attention_mask,
                do_sample=True,
                max_length=256,
                num_beams=2,
                top_k=80,
                top_p=0.90,
                early_stopping=True,
                num_return_sequences=1
            ).to(self.device_name)
            return self.simplifier_tokenizer.decode(
                beam_outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
            )

        pred_sents = [generate(source) for source in batch["source"]]
        score = corpus_sari(batch["source"], pred_sents, [batch["targets"]])
        loss = 1 - score / 100
        save_log(
            self.training_parameters['output_dir'],
            self.training_parameters['model_name'],
            self.current_epoch,
            loss=loss,
            sari=score,
            data_type='validation'
        )
        return loss

    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler.

        Returns:
            list: A list containing the optimizer and scheduler.
        """
        model1 = self.summarizer
        model2 = self.simplifier
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model2.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.training_parameters['weight_decay'],
            },
            {
                "params": [p for n, p in model2.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in model1.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.training_parameters['weight_decay'],
            },
            {
                "params": [p for n, p in model1.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": self.W
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.training_parameters['learning_rate'], eps=self.training_parameters['adam_epsilon'])
        t_total = (
                (
                        len(self.train_dataloader().dataset) // self.training_parameters['train_batch_size']
                ) // self.training_parameters['gradient_accumulation_steps']
                * float(self.training_parameters['num_train_epochs'])
        )

        if self.training_parameters['scheduler_type'] == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=self.training_parameters['warmup_steps'], num_training_steps=t_total
            )
        else:
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=self.training_parameters['warmup_steps'], num_training_steps=t_total
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
            tokenizer=self.simplifier_tokenizer,
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
            tokenizer=self.simplifier_tokenizer,
            max_len=self.max_seq_length,
            sample_size=self.valid_sample_size
        )
        return DataLoader(
            val_dataset,
            batch_size=self.valid_batch_size
        )
