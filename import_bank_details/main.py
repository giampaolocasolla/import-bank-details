import os
import glob
import yaml
import pandas as pd
import logging
from logging.handlers import RotatingFileHandler


def setup_logging():
    """
    Set up logging configuration to include timestamps and display logs in the terminal.
    A rotating file handler is also used to limit the log file size and keep backups.
    """
    global logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Create handlers - one for writing to log files and another for streaming to the console
    # 1MB per file, keeping 5 backups
    file_handler = RotatingFileHandler(
        'app.log', maxBytes=1024*1024, backupCount=5)
    stream_handler = logging.StreamHandler()

    # Define the logging format to include timestamps
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)


def load_config(config_path):
    """
    Load the configuration from the YAML file.

    Args:
        config_path (str): Path to the configuration YAML file.

    Returns:
        dict: Configuration dictionary.
    """
    with open(config_path, 'r', encoding='utf-8') as stream:
        try:
            config = yaml.safe_load(stream)
            logger.info('Configuration file loaded successfully.')
            return config
        except yaml.YAMLError as exc:
            logger.error('Error loading configuration file:', exc)
            raise


def get_latest_files(data_dir):
    """
    Identify the most recently modified file in each data subfolder to process.

    Args:
        data_dir (str): Path to the data directory.

    Returns:
        dict: Dictionary with folder names as keys and latest file paths as values.
    """
    cwd = os.getcwd()
    folders_data = os.listdir(data_dir)
    file_data = {folder: max(glob.glob(os.path.join(cwd, data_dir, folder, '*')), key=os.path.getctime)
                 for folder in folders_data}
    logger.info('Latest files from data folders obtained.')
    return file_data


def import_data(file_path, import_params=None):
    """
    Import data from a file with specified parameters.

    Args:
        file_path (str): Path to the data file.
        import_params (dict, optional): Parameters for importing the data. Defaults to None.

    Returns:
        pd.DataFrame: Imported data as a DataFrame.
    """
    try:
        if import_params:
            df = pd.read_csv(file_path, **import_params)
            logger.info(f'Data imported with parameters from {file_path}.')
        else:
            df = pd.read_csv(file_path)
            logger.info(f'Data imported without parameters from {file_path}.')
    except UnicodeDecodeError:
        df = pd.read_excel(file_path)
        logger.info(f'Data imported as Excel from {file_path}.')
    except Exception as e:
        logger.error(f'Error importing data from {file_path}: {e}')
        raise
    return df


def process_data(df, config, bank_name):
    """
    Process and clean the data according to the configuration.

    Args:
        df (pd.DataFrame): Data to be processed.
        config (dict): Configuration dictionary for the bank.
        bank_name (str): Name of the bank.

    Returns:
        pd.DataFrame: Processed data as a DataFrame.
    """
    # Assign the folder name to the 'Bank' column for identification
    if 'Bank' in config['columns_old']:
        df['Bank'] = bank_name
        logger.info(f'Bank column set for {bank_name}.')

    # Select and rename columns as per the new configuration mapping
    df = df[config['columns_old']]
    df = df.rename(columns=dict(
        zip(config['columns_old'], config['columns_new'])))
    logger.info(f'Columns selected and renamed for {bank_name}.')

    # Call remove_unnecessary_expenses if the 'Remove' key exists in config
    if 'Remove' in config:
        df = remove_unnecessary_expenses(df, config['Remove'])
        logger.info(f'Unnecessary expenses removed for {bank_name}.')

    # Parse the 'Day' column into datetime format as specified in the configuration
    df['Day'] = pd.to_datetime(df['Day'], format=config['Day'])
    logger.info(f'Day column converted to datetime for {bank_name}.')

    return df


def remove_unnecessary_expenses(df, remove_criteria):
    """
    Adjust the dataframe by removing rows based on 'remove_criteria' in 'Expense name' column,
    replacing NaN values with an empty string to avoid TypeError with bitwise NOT operator.

    Args:
        df (pd.DataFrame): DataFrame to be filtered.
        remove_criteria (list): List of strings to be removed from the 'Expense name' column.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    # Replace NaN values with an empty string and ensure that 'Expense name' is a string
    df['Expense name'] = df['Expense name'].fillna('').astype(str)

    # Use str.contains to create a boolean mask, while setting na parameter to False
    # to handle NaN values appropriately
    mask = df['Expense name'].str.contains('|'.join(remove_criteria), na=False)

    # Apply the mask using the bitwise NOT operator to filter the dataframe
    return df[~mask]


def save_to_excel(df, output_dir, folders_data):
    """
    Save the processed data to an Excel file in the output directory.

    Args:
        df (pd.DataFrame): DataFrame to be saved.
        output_dir (str): Path to the output directory.
        folders_data (list): List of folder names to be included in the filename.
    """
    # Create a filename for the output file based on the latest date in the data and the folders_data
    filename = f"{df['Day'].max().strftime('%Y-%m-%d')}_{'-'.join(folders_data)}.xlsx"

    # Format the 'Day' column for final output
    df['Day'] = df['Day'].dt.strftime('%d/%m/%Y')

    # Write the processed data to an Excel file in the output directory
    df.to_excel(os.path.join(output_dir, filename), index=False)
    logger.info(f'Dataframe saved to Excel file {filename}.')


def main():
    """Main function to orchestrate the data import, processing, and export."""
    # Set up logging
    setup_logging()

    # Load the configuration from the YAML file
    config = load_config("config.yaml")

    # Identify the most recently modified file in each data subfolder to process
    file_data = get_latest_files('data')

    # Initialize the dataframe that will hold all the data
    df = None

    # Iterate through each file, process and clean the data according to the configuration
    for bank_name, file_path in file_data.items():
        try:
            # Import data with specified parameters if any, otherwise default to a basic read
            df_temp = import_data(file_path, config[bank_name].get('import'))

            # Process and clean the data
            df_temp = process_data(df_temp, config[bank_name], bank_name)

            # Combine the current file's data with the main dataframe
            df = pd.concat([df, df_temp], ignore_index=True)
        except Exception as e:
            logger.error(f'Error processing data for {bank_name}: {e}')
            continue

    if df is not None:
        # Round down the 'Day' column to the nearest day
        df['Day'] = df['Day'].dt.floor('D')

        # Sort the combined data for uniformity and easier analysis
        df = df.sort_values(by=['Day', 'Expense name', 'Amount'])

        # Convert the 'Amount' to a negative value to indicate expense
        df['Amount'] = -df['Amount']

        # Save the processed data to an Excel file in the output directory
        save_to_excel(df, 'output', file_data.keys())


if __name__ == '__main__':
    main()
