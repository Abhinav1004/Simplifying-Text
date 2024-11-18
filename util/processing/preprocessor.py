# Import required libraries
from pathlib import Path
import hashlib


def yield_lines(filepath):
    """
    Generator function to yield lines from a file.

    Args:
        filepath (str or Path): Path to the file.

    Yields:
        str: Each line from the file, stripped of trailing whitespace.
    """
    filepath = Path(filepath)
    with filepath.open('r') as f:
        for line in f:
            yield line.rstrip()


def read_lines(filepath):
    """
    Reads all lines from a file and returns them as a list.

    Args:
        filepath (str or Path): Path to the file.

    Returns:
        list: List of lines from the file, each stripped of trailing whitespace.
    """
    return [line.rstrip() for line in yield_lines(filepath)]


def get_data_filepath(data_set_dir, dataset, phase, data_type, i=None):
    """
    Constructs the file path for a dataset file based on provided parameters.

    Args:
        data_set_dir (str or Path): Directory containing datasets.
        dataset (str): Name of the dataset.
        phase (str): Phase of the data (e.g., 'train' or 'valid').
        data_type (str): Type of data (e.g., 'complex' or 'simple').
        i (int, optional): Optional index to append as a suffix to the filename.

    Returns:
        Path: Constructed file path as a Path object.
    """
    suffix = f'.{i}' if i is not None else ''
    data_filename = f'{dataset}.{phase}.{data_type}{suffix}'
    return Path(data_set_dir) / dataset / data_filename


def generate_hash(data):
    h = hashlib.new('md5')
    h.update(str(data).encode())
    return h.hexdigest()


def count_line(filepath):
    filepath = Path(filepath)
    line_count = 0
    with filepath.open("r") as f:
        for line in f:
            line_count += 1
    return line_count


def write_lines(lines, filepath):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w") as fout:
        for line in lines:
            fout.write(line + '\n')
