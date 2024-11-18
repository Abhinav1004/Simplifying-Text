# Import necessary libraries
from nltk.tokenize import word_tokenize
from pathlib import Path
import textstat
from util.processing.preprocessor import get_data_filepath
from easse.sari import corpus_sari as easse_corpus_sari
from easse.fkgl import corpus_fkgl as easse_corpus_fkgl


def load_dataset(dataset_dir, dataset_name, phase='test'):
    """
    Load the dataset for evaluation.

    Args:
        dataset_dir (str or Path): Path to the dataset directory.
        dataset_name (str): Name of the dataset (e.g., 'dwiki' or 'wiki_doc').
        phase (str): Dataset phase to load ('train', 'valid', 'test').

    Returns:
        tuple: (list of complex sentences, list of simple sentences)
    """
    complex_filepath = get_data_filepath(dataset_dir, dataset_name, phase, 'complex')
    simple_filepath = get_data_filepath(dataset_dir, dataset_name, phase, 'simple')

    # Read lines from files
    complex_sents = Path(complex_filepath).read_text().splitlines()
    simple_sents = Path(simple_filepath).read_text().splitlines()

    return complex_sents, simple_sents


class BartModelEvaluator:
    """
    A class for evaluating a BART-based summarization model using SARI, D-SARI, and FKGL metrics.

    Args:
        model_config : Configuration dictionary containing the device to run the model on ("cuda", "cpu", or "mps").
        model (BartForConditionalGeneration): Pre-trained BART model to evaluate.
        tokenizer (BartTokenizer): Tokenizer for the BART model.
    """
    def __init__(self, model_config, model, tokenizer):
        self.model = model.to(model_config['device'])
        self.tokenizer = tokenizer
        self.device = model_config['device']
        self.max_seq_length = model_config['max_seq_length']

    def generate_summary(self, sentence, max_length=256):
        """
        Generate a summary for a given input sentence using the BART model.

        Args:
            sentence (str): Input sentence to be summarized.
            max_length (int): Maximum length of the generated summary.

        Returns:
            str: Generated summary.
        """
        inputs = self.tokenizer(
            sentence,
            return_tensors="pt",
            max_length=self.max_seq_length,
            truncation=True,
            padding="max_length"
        ).to(self.device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        summary_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=5,
            early_stopping=True
        )

        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

    @staticmethod
    def calculate_sari_and_d_sari(source_sent, predicted_sent, references):
        """
        Calculate SARI and D-SARI scores for text simplification.

        Args:
            source_sent (str): Source sentence.
            predicted_sent (str): Predicted simplified sentence.
            references (list): List of reference simplified sentences.

        Returns:
            tuple: (SARI score, D-SARI score)
        """
        source_tokens = set(word_tokenize(source_sent))
        predicted_tokens = set(word_tokenize(predicted_sent))
        reference_tokens = [set(word_tokenize(ref)) for ref in references]

        # Calculate addition, deletion, and keep scores
        add_scores = [
            len(predicted_tokens - ref) / max(1, len(predicted_tokens))
            for ref in reference_tokens
        ]
        keep_scores = [
            len(predicted_tokens & ref) / max(1, len(ref))
            for ref in reference_tokens
        ]
        delete_score = len(source_tokens - predicted_tokens) / max(1, len(source_tokens))

        sari = (sum(add_scores) + sum(keep_scores) + delete_score) / (len(add_scores) + len(keep_scores) + 1)
        d_sari = delete_score  # D-SARI focuses specifically on the deletion component

        return sari, d_sari

    def calculate_fkgl(self, text):
        """
        Calculate the Flesch-Kincaid Grade Level (FKGL) score.

        Args:
            text (str): Input text.

        Returns:
            float: FKGL score.
        """
        return textstat.flesch_kincaid_grade(text)

    def evaluate(self, source_sentences, reference_sentences):
        """
        Evaluate a set of source and reference sentences using SARI, D-SARI, and FKGL metrics.

        Args:
            source_sentences (list): List of source sentences to be simplified.
            reference_sentences (list): List of corresponding reference sentences.

        Returns:
            dict: Dictionary containing average SARI, D-SARI, and FKGL scores.
        """
        total_sari, total_d_sari, total_fkgl = 0, 0, 0
        predictions = []

        for i, source_sent in enumerate(source_sentences):
            try:
                predicted_sent = self.generate_summary(source_sent)
            except Exception as e:
                print(f"Error generating summary for sample {i}: {e}")
                predicted_sent = ""  # Fallback to an empty prediction

            predictions.append(predicted_sent)
            references = [reference_sentences[i]]  # Assuming one reference per source

            # Calculate SARI and D-SARI scores
            sari, d_sari = self.calculate_sari_and_d_sari(source_sent, predicted_sent, references)
            total_sari += sari
            total_d_sari += d_sari

            # Calculate FKGL score
            fkgl = self.calculate_fkgl(predicted_sent)
            total_fkgl += fkgl

            # Calculate EASSE SARI and FKGL for this sample
            try:
                easse_sari = easse_corpus_sari(orig_sents=[source_sent], sys_sents=[predicted_sent], refs_sents=[references])
                easse_fkgl = easse_corpus_fkgl([predicted_sent])
            except Exception as e:
                print(f"Error calculating EASSE metrics for sample {i}: {e}")
                easse_sari = 0
                easse_fkgl = 0

            print(f"Sample {i + 1}/{len(source_sentences)}")
            print(f"Source: {source_sent}")
            print(f"Predicted: {predicted_sent}")
            print(f"Reference: {references[0]}")
            print(f"SARI: {sari:.2f}, D-SARI: {d_sari:.2f}, FKGL: {fkgl:.2f}")
            print(f"EASSE SARI: {easse_sari:.2f}, EASSE FKGL: {easse_fkgl:.2f}\n")

        # Calculate average scores
        avg_sari = total_sari / len(source_sentences)
        avg_d_sari = total_d_sari / len(source_sentences)
        avg_fkgl = total_fkgl / len(source_sentences)

        # Calculate EASSE SARI and FKGL scores for all predictions
        try:
            easse_sari = easse_corpus_sari(orig_sents=source_sentences, sys_sents=predictions,
                                           refs_sents=[reference_sentences])
            easse_fkgl = easse_corpus_fkgl(predictions)
        except Exception as e:
            print(f"Error calculating EASSE metrics for all predictions: {e}")
            easse_sari = 0
            easse_fkgl = 0

        print(f"Average SARI: {avg_sari:.2f}")
        print(f"Average D-SARI: {avg_d_sari:.2f}")
        print(f"Average FKGL: {avg_fkgl:.2f}")
        print(f"EASSE SARI: {easse_sari:.2f}")
        print(f"EASSE FKGL: {easse_fkgl:.2f}")

        return {
            "SARI": avg_sari,
            "D-SARI": avg_d_sari,
            "FKGL": avg_fkgl,
            "EASSE SARI": easse_sari,
            "EASSE FKGL": easse_fkgl
        }
