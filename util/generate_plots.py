# Import relevant libraries
import pandas as pd
import matplotlib.pyplot as plt
import os


def identify_files(folder_path):
    """
    Identifies and categorizes files in a specified folder based on
    substrings "_training_log", "_validation_log", and "_evaluation_metrics".

    Parameters:
    - folder_path (str): Path to the folder containing files.

    Returns:
    - dict: A dictionary with categorized files, with keys:
      'training_log', 'validation_log', and 'evaluation_metrics'.
    """
    categorized_files = {
        'training_log': None,
        'validation_log': None,
        'evaluation_metrics': None
    }

    # Iterate over files in the folder
    for file_name in os.listdir(folder_path):
        if "_training_log" in file_name:
            categorized_files['training_log'] = os.path.join(folder_path, file_name)
        elif "_validation_log" in file_name:
            categorized_files['validation_log'] = os.path.join(folder_path, file_name)
        elif "_evaluation_metrics" in file_name:
            categorized_files['evaluation_metrics'] = os.path.join(folder_path, file_name)

    return categorized_files


def plot_average_loss(training_log_path, validation_log_path, output_file='average_loss.csv'):
    """
    Plots the average training and validation loss over epochs, and saves
    the averaged loss data to a CSV file.

    Parameters:
    - training_log_path (str): Path to the training log CSV file.
    - validation_log_path (str): Path to the validation log CSV file.
    - output_file (str): Path to save the output CSV file containing average loss data.
    """
    # Load data
    training_log = pd.read_csv(training_log_path)
    validation_log = pd.read_csv(validation_log_path)

    # Calculate average loss
    avg_training_loss = training_log.groupby('epoch')['loss'].mean().reset_index()
    avg_training_loss['data_type'] = 'training'
    avg_validation_loss = validation_log.groupby('epoch')['loss'].mean().reset_index()
    avg_validation_loss['data_type'] = 'validation'

    # Combine the data
    combined_loss = pd.concat([avg_training_loss, avg_validation_loss], axis=0)
    combined_loss.columns = ['epoch', 'average_loss', 'data_type']

    # Save to CSV
    combined_loss.to_csv(output_file, index=False)

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(avg_training_loss['epoch'], avg_training_loss['average_loss'], marker='o', linestyle='-', color='b', label='Average Training Loss')
    plt.plot(avg_validation_loss['epoch'], avg_validation_loss['average_loss'], marker='x', linestyle='--', color='r', label='Average Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Epoch vs Average Loss (Training and Validation)')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_metric_distributions(evaluation_metrics_path, output_file='average_metrics.csv'):
    """
    Plots the distribution of specified metrics from the evaluation metrics data
    and saves the averaged metrics to a CSV file.

    Parameters:
    - evaluation_metrics_path (str): Path to the evaluation metrics CSV file.
    - output_file (str): Path to save the output CSV file containing average metrics data.
    """
    # Load data
    evaluation_metrics = pd.read_csv(evaluation_metrics_path)
    metrics_columns = ['SARI', 'D-SARI', 'FKGL', 'EASSE SARI', 'EASSE FKGL']

    # Calculate average metrics
    avg_metrics = evaluation_metrics[metrics_columns].mean().reset_index()
    avg_metrics.columns = ['metric', 'average_value']
    avg_metrics['data_type'] = 'evaluation'

    # Save to CSV
    avg_metrics.to_csv(output_file, index=False)

    # Plot distributions
    for metric in metrics_columns:
        plt.figure(figsize=(8, 5))
        evaluation_metrics[metric].plot(kind='hist', bins=30, alpha=0.7, color='teal', edgecolor='black')
        plt.xlabel(metric)
        plt.ylabel('Frequency')
        plt.title(f'Distribution of {metric}')
        plt.grid(True)
        plt.show()
