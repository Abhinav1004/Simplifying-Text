from util.evaluate_model.sari import corpus_sari
from util.preprocessing.preprocessor import OUTPUT_DIR
from argparse import ArgumentParser
import nltk

nltk.download('punkt')
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import (
    AdamW, BartForConditionalGeneration, BartTokenizerFast, get_cosine_schedule_with_warmup
)
from Ts_BART import BartFineTuner


class SumSim(pl.LightningModule):
    def __init__(self, args):
        super(SumSim, self).__init__()
        self.args = args
        self.save_hyperparameters()
        # Load pre-trained model and tokenizer
        # self.summarizer = BartModel.from_pretrained("facebook/bart-large-cnn")
        self.summarizer = BartForConditionalGeneration.from_pretrained(self.args.sum_model)
        self.summarizer_tokenizer = BartTokenizerFast.from_pretrained(self.args.sum_model)
        self.summarizer = self.summarizer.to(self.args.device)

        # self.simplifier = BartForConditionalGeneration.from_pretrained(self.args.sum_model)
        self.simplifier = BartFineTuner.load_from_checkpoint("experiments/exp_WikiLarge_BARTSingle/checkpoint-epoch=2.ckpt")
        self.simplifier = self.simplifier.model.to(self.args.device)
        self.simplifier_tokenizer = BartTokenizerFast.from_pretrained(self.args.sim_model)

    def is_logger(self):
        return self.trainer.global_rank <= 0

    def forward(self, input_ids,
                attention_mask=None,
                decoder_input_ids=None,
                decoder_attention_mask=None, labels=None):

        outputs = self.simplifier(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels
        )

        return outputs

    def training_step(self, batch, batch_idx):
        source = batch["source"]
        labels = batch['target_ids']
        labels[labels[:, :] == self.simplifier_tokenizer.pad_token_id] = -100
        # zero the gradient buffers of all parameters
        self.opt.zero_grad()
        # print(source, len(source))
        ## summarizer stage
        inputs = self.summarizer_tokenizer(
            source,
            max_length=512,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        ).to(self.args.device)

        src_ids = inputs['input_ids'].to(self.args.device)
        src_mask = inputs['attention_mask'].to(self.args.device)

        # compute the loss between summarization and simplification target
        # sum_outputs.loss

        sum_outputs = self.summarizer(
            input_ids=src_ids,
            attention_mask=src_mask,
            labels=labels,
            decoder_attention_mask=batch['target_mask']
        )

        # H1 = sum_outputs.encoder_last_hidden_state

        # generate summary
        summary_ids = self.summarizer.generate(
            inputs['input_ids'].to(self.args.device),
            num_beams=5,
            min_length=10,
            max_length=256,
            top_k=120, top_p=0.95,
        ).to(self.args.device)

        ### Original loss
        # summary_attention_mask = torch.ones(summary_ids.shape).to(self.args.device)
        # summary_attention_mask[summary_ids[:,:]==self.summarizer_tokenizer.pad_token_id]=0

        # sim_outputs  = self(
        #     input_ids = summary_ids,
        #     attention_mask = summary_attention_mask,
        #     labels = labels,
        #     decoder_attention_mask = batch['target_mask']
        # )

        ### modified loss
        padded_summary_ids = torch.zeros((summary_ids.shape[0], 256), dtype=torch.long).fill_(
            self.simplifier_tokenizer.pad_token_id).to(self.args.device)

        for i, summary_id in enumerate(summary_ids):
            padded_summary_ids[i, :summary_id.shape[0]] = summary_id

        summary_attention_mask = torch.ones(padded_summary_ids.shape).to(self.args.device)
        summary_attention_mask[padded_summary_ids[:, :] == self.summarizer_tokenizer.pad_token_id] = 0

        # forward pass
        sim_outputs = self(
            input_ids=padded_summary_ids,
            attention_mask=summary_attention_mask,
            labels=labels,
            decoder_attention_mask=batch['target_mask']
        )
        # H2 = sim_outputs.encoder_last_hidden_state

        ## CosSim
        # Rep1 = torch.matmul(H1, self.W)
        # Rep2 = torch.matmul(H2, self.W)
        # Rep1 = self.relu(Rep1)
        # Rep2 = self.relu(Rep2)
        # CosSim = nn.CosineSimilarity(dim=2, eps=1e-6)
        # sim_score = CosSim(Rep1, Rep2)

        ## KL loss
        # H1 = torch.transpose((torch.transpose(H1, 1,2)@self.Q), 1,2)
        # H2 = torch.transpose((torch.transpose(H2, 1,2)@self.Q), 1,2)
        # Rep1 = torch.matmul(H1, self.W)
        # Rep2 = torch.matmul(H2, self.W)
        # Rep1 = Rep1.squeeze(dim=2)
        # Rep2 = Rep2.squeeze(dim=2)
        # LogSoftMax = nn.LogSoftmax(dim=1)
        # Rep1 = LogSoftMax(Rep1)
        # Rep2 = LogSoftMax(Rep2)

        if self.args.custom_loss:
            '''
            Custom Loss:
            Loss = oiginal_loss + lambda**2 * complexity_score

            - ratio: control the ratio of sentences we want to compute complexity for training.
            - lambda: control the weight of the complexity loss.
            '''
            loss = sim_outputs.loss * self.args.w1
            # loss += sum_outputs.loss * self.args.w2
            ### KL ###
            # loss += (self.args.lambda_ * self.kl_loss(Rep1, Rep2))

            ### CosSim ###
            # loss += (-self.args.lambda_ * (sim_score.mean(dim=1).mean(dim=0)))

            # self.manual_backward(loss)
            # self.opt.step()

            self.log('train_loss', sim_outputs.loss, on_step=True, prog_bar=True, logger=True)
            # print(loss)
            return loss
        else:
            loss = sim_outputs.loss
            self.log('train_loss', loss, on_step=True, prog_bar=True, logger=True)
            # print(loss)
            return loss

    def validation_step(self, batch, batch_idx):
        loss = self.sari_validation_step(batch)
        # loss = self._step(batch)
        print("Val_loss", loss)
        logs = {"val_loss": loss}

        self.log('val_loss', loss, batch_size=self.args.valid_batch_size)
        return torch.tensor(loss, dtype=float)

    def sari_validation_step(self, batch):
        def generate(sentence):
            text = sentence
            # summarize the document
            inputs = self.summarizer_tokenizer(
                [text],
                max_length=512,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            # generate summary
            summary_ids = self.summarizer.generate(
                inputs['input_ids'].to(self.args.device),
                num_beams=5,
                min_length=10,
                max_length=256,
                top_k=120, top_p=0.95,
            ).to(self.args.device)

            summary_attention_mask = torch.ones(summary_ids.shape).to(self.args.device)
            summary_attention_mask[summary_ids == self.summarizer_tokenizer.pad_token_id] = 0

            # set top_k = 130 and set top_p = 0.95 and num_return_sequences = 1
            beam_outputs = self.simplifier.generate(
                input_ids=summary_ids,
                attention_mask=summary_attention_mask,
                do_sample=True,
                max_length=self.args.max_seq_length,
                num_beams=5,
                top_k=130,
                top_p=0.95,
                early_stopping=True,
                num_return_sequences=1
            ).to(self.device)
            # final_outputs = []
            # for beam_output in beam_outputs:

            ## Bart:
            sent = self.simplifier_tokenizer.decode(beam_outputs[0], skip_special_tokens=True,
                                                    clean_up_tokenization_spaces=True)

            # if sent.lower() != sentence.lower() and sent not in final_outputs:
            # final_outputs.append(sent)

            return sent

        pred_sents = []
        for source in batch["source"]:
            pred_sent = generate(source)
            pred_sents.append(pred_sent)

        ### WIKI-large ###
        score = corpus_sari(batch["source"], pred_sents, [batch["targets"]])

        ### turkcorpuse ###
        # score = corpus_sari(batch["source"], pred_sents, batch["targets"])

        print("Sari score: ", score)

        return 1 - score / 100

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        model1 = self.summarizer
        model2 = self.simplifier
        # no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model1.named_parameters()]
            },
            {
                "params": [p for n, p in model2.named_parameters()]
            },
            # {
            #     "params": self.W
            # },
            # {
            #     "params": self.Q
            # },
            # {
            #     "params": self.W2
            # }
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(self, epoch=None, batch_idx=None, optimizer=None, optimizer_idx=None, optimizer_closure=None,
                       on_tpu=None, using_native_amp=None, using_lbfgs=None):
        optimizer.step(closure=optimizer_closure)

        optimizer.zero_grad()
        self.lr_scheduler.step()

    def save_core_model(self):
        tmp = self.args.model_name + 'core'
        store_path = OUTPUT_DIR / tmp
        self.model.save_pretrained(store_path)
        self.simplifier_tokenizer.save_pretrained(store_path)

    def train_dataloader(self):
        train_dataset = TrainDataset(dataset=self.args.dataset,
                                     tokenizer=self.simplifier_tokenizer,
                                     max_len=self.args.max_seq_length,
                                     sample_size=self.args.train_sample_size)

        dataloader = DataLoader(train_dataset,
                                batch_size=self.args.train_batch_size,
                                drop_last=True,
                                shuffle=True,
                                pin_memory=True,
                                num_workers=4)
        t_total = ((len(dataloader.dataset) // (self.args.train_batch_size * max(1, self.args.n_gpu)))
                   // self.args.gradient_accumulation_steps
                   * float(self.args.num_train_epochs)
                   )
        # scheduler = get_linear_schedule_with_warmup(
        #     self.opt, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total
        # )
        scheduler = get_cosine_schedule_with_warmup(
            self.opt, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        val_dataset = ValDataset(dataset=self.args.dataset,
                                 tokenizer=self.simplifier_tokenizer,
                                 max_len=self.args.max_seq_length,
                                 sample_size=self.args.valid_sample_size)
        return DataLoader(val_dataset,
                          batch_size=self.args.valid_batch_size,
                          num_workers=4)

    @staticmethod
    def add_model_specific_args(parent_parser):
        p = ArgumentParser(parents=[parent_parser], add_help=False)
        # facebook/bart-base
        p.add_argument('-HiddenSize', '--hidden_size', type=int, default=1)
        p.add_argument('-SeqDim', '--seq_dim', type=int, default=512)
        p.add_argument('-Weight1', '--w1', type=int, default=1)
        p.add_argument('-Weight2', '--w2', type=int, default=1)
        p.add_argument('-Lambda', '--lambda_', type=int, default=11)
        # BRIO: Yale-LILY/brio-cnndm-uncased ainize/bart-base-cnn
        p.add_argument('-Summarizer', '--sum_model', default='ainize/bart-base-cnn')
        p.add_argument('-Simplifier', '--sim_model', default='facebook/bart-base')
        p.add_argument('-TrainBS', '--train_batch_size', type=int, default=6)
        p.add_argument('-ValidBS', '--valid_batch_size', type=int, default=6)
        p.add_argument('-lr', '--learning_rate', type=float, default=5e-5)
        p.add_argument('-MaxSeqLen', '--max_seq_length', type=int, default=256)
        p.add_argument('-AdamEps', '--adam_epsilon', default=1e-8)
        p.add_argument('-WeightDecay', '--weight_decay', default=0.0001)
        p.add_argument('-WarmupSteps', '--warmup_steps', default=5)
        p.add_argument('-NumEpoch', '--num_train_epochs', default=7)
        p.add_argument('-CosLoss', '--custom_loss', default=False)
        p.add_argument('-GradAccuSteps', '--gradient_accumulation_steps', default=1)
        p.add_argument('-GPUs', '--n_gpu', default=torch.cuda.device_count())
        p.add_argument('-nbSVS', '--nb_sanity_val_steps', default=-1)
        p.add_argument('-TrainSampleSize', '--train_sample_size', default=1)
        p.add_argument('-ValidSampleSize', '--valid_sample_size', default=1)
        p.add_argument('-device', '--device', default='cuda')
        # p.add_argument('-NumBeams','--num_beams', default=8)
        return p
