# Import libraries
from torch.utils.data import Dataset
from util.processing.preprocessor import (
    yield_lines, read_lines, get_data_filepath
)


class TrainDataset(Dataset):
    def __init__(self, data_set_dir, dataset, tokenizer, max_len=256, sample_size=1):
        """
        Initializes the training dataset.

        Args:
            data_set_dir: Path to data
            dataset: Name of the dataset.
            tokenizer: Tokenizer object to tokenize the data.
            max_len (int): Maximum length of the tokenized sequences.
            sample_size (float): Fraction of the dataset to sample.
        """
        self.sample_size = sample_size
        self.max_len = max_len
        self.tokenizer = tokenizer

        print("Initializing TrainDataset...")
        self.source_filepath = get_data_filepath(data_set_dir, dataset, 'train', 'complex')
        self.target_filepath = get_data_filepath(data_set_dir, dataset, 'train', 'simple')
        print("Dataset paths initialized.")

        self._load_data()

    def _load_data(self):
        """Loads the source and target data."""
        self.inputs = read_lines(self.source_filepath)
        self.targets = read_lines(self.target_filepath)

    def __len__(self):
        """Returns the length of the dataset based on the sample size."""
        return int(len(self.inputs) * self.sample_size)

    def __getitem__(self, index):
        """Fetches a single item from the dataset."""
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
        src_mask = tokenized_inputs["attention_mask"].squeeze()
        target_mask = tokenized_targets["attention_mask"].squeeze()

        return {
            "source_ids": source_ids, 
            "source_mask": src_mask, 
            "target_ids": target_ids, 
            "target_mask": target_mask,
            "sources": source, 
            "targets": [target],
            "source": source, 
            "target": target
        }


class ValDataset(Dataset):
    def __init__(self, data_set_dir, dataset, tokenizer, max_len=256, sample_size=1):
        """
        Initializes the validation dataset.

        Args:
            data_set_dir: Path to data
            dataset: Name or path of the dataset.
            tokenizer: Tokenizer object to tokenize the data.
            max_len (int): Maximum length of the tokenized sequences.
            sample_size (float): Fraction of the dataset to sample.
        """
        self.sample_size = sample_size
        self.max_len = max_len
        self.tokenizer = tokenizer

        print("Initializing ValDataset...")
        self.source_filepath = get_data_filepath(data_set_dir, dataset, 'valid', 'complex')
        self.target_filepaths = get_data_filepath(data_set_dir, dataset, 'valid', 'simple')
        print("Dataset paths initialized.")

        self._load_data()

    def _load_data(self):
        """Loads the source and target data."""
        self.inputs = [line for line in yield_lines(self.source_filepath)]
        self.targets = [line for line in yield_lines(self.target_filepaths)]

    def __len__(self):
        """Returns the length of the dataset based on the sample size."""
        return int(len(self.inputs) * self.sample_size)

    def __getitem__(self, index):
        """Fetches a single item from the dataset."""
        return {
            "source": self.inputs[index], 
            "targets": self.targets[index]
        }
