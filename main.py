import os
import glob
import yaml
import pandas as pd
import logging
from logging.handlers import RotatingFileHandler

# Set up logging to include timestamps and display logs in the terminal
# A rotating file handler is also used to limit the log file size and keep backups

# Define a custom logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create handlers - one for writing to log files and another for streaming to the console
file_handler = RotatingFileHandler('app.log', maxBytes=1024*1024, backupCount=5)  # 1MB per file, keeping 5 backups
stream_handler = logging.StreamHandler()

# Define the logging format to include timestamps
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                              datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

def remove_unnecessary_expenses(df, remove_criteria):
    """
    Adjusts the dataframe by removing rows based on 'remove_criteria' in 'Expense name' column,
    replacing NaN values with an empty string to avoid TypeError with bitwise NOT operator.
    """
    # Replace NaN values with an empty string and ensure that 'Expense name' is a string
    df['Expense name'] = df['Expense name'].fillna('').astype(str)

    # Use str.contains to create a boolean mask, while setting na parameter to False
    # to handle NaN values appropriately
    mask = df['Expense name'].str.contains('|'.join(remove_criteria), na=False)

    # Apply the mask using the bitwise NOT operator to filter the dataframe
    return df[~mask]

def main():
    # Get the current working directory to define file paths
    cwd = os.getcwd()
    logger.info('Current working directory obtained.')

    # Load the configuration from the YAML file to set up file processing
    with open("config.yaml", 'r', encoding='utf-8') as stream:
        try:
            config = yaml.safe_load(stream)
            logger.info('Configuration file loaded successfully.')
        except yaml.YAMLError as exc:
            logger.error('Error loading configuration file:', exc)
            return

    # Identify the most recently modified file in each data subfolder to process
    folders_data = os.listdir('data')
    file_data = {folder: max(glob.glob(os.path.join(cwd, 'data', folder, '*')), key=os.path.getctime) 
                 for folder in folders_data}
    logger.info('Latest files from data folders obtained.')

    # Initialize the dataframe that will hold all the data
    df = None

    # Iterate through each file, process and clean the data according to the configuration
    for file_key, file_value in file_data.items():
        try:
            # Import data with specified parameters if any, otherwise default to a basic read
            if config[file_key]['import']:
                df_temp = pd.read_csv(file_value, **config[file_key]['import'])
                logger.info(f'Data imported with parameters for {file_key}.')
            else:
                df_temp = pd.read_csv(file_value)
                logger.info(f'Data imported without parameters for {file_key}.')
        except UnicodeDecodeError:
            df_temp = pd.read_excel(file_value)
            logger.info(f'Data imported as Excel for {file_key}.')
        except Exception as e:
            logger.error(f'Error importing data for {file_key}: {e}')
            continue

        # Assign the folder name to the 'Bank' column for identification
        if 'Bank' in config[file_key]['columns_old']:
            df_temp['Bank'] = file_key
            logger.info(f'Bank column set for {file_key}.')

        # Select and rename columns as per the new configuration mapping
        df_temp = df_temp[config[file_key]['columns_old']]
        df_temp = df_temp.rename(columns=dict(zip(config[file_key]['columns_old'], config[file_key]['columns_new'])))
        logger.info(f'Columns selected and renamed for {file_key}.')

        # Call remove_unnecessary_expenses if the 'Remove' key exists in config
        if 'Remove' in config[file_key]:
            df_temp = remove_unnecessary_expenses(df_temp, config[file_key]['Remove'])
            logger.info(f'Unnecessary expenses removed for {file_key}.')

        # Parse the 'Day' column into datetime format as specified in the configuration
        df_temp['Day'] = pd.to_datetime(df_temp['Day'], format=config[file_key]['Day'])
        logger.info(f'Day column converted to datetime for {file_key}.')

        # Combine the current file's data with the main dataframe
        df = pd.concat([df, df_temp], ignore_index=True)
    df['Day'] = df['Day'].dt.floor('D')

    # Sort the combined data for uniformity and easier analysis
    df = df.sort_values(by=['Day', 'Expense name', 'Amount'])

    # Convert the 'Amount' to a negative value to indicate expense
    df['Amount'] = -df['Amount']

    # Create a filename for the output file based on the latest date in the data and the folders_data
    filename = f"{df['Day'].max().strftime('%Y-%m-%d')}_{'-'.join(folders_data)}.xlsx"
    
    # Format the 'Day' column for final output
    df['Day'] = df['Day'].dt.strftime('%d/%m/%Y')

    # Write the processed data to an Excel file in the output directory
    df.to_excel(os.path.join('output', filename), index=False)
    logger.info(f'Dataframe saved to Excel file {filename}.')

if __name__ == '__main__':
    main()