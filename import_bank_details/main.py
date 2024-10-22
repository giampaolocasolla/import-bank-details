import glob
import logging
import os
from typing import Dict, List, Optional

import pandas as pd

from import_bank_details.classification import classify_expenses
from import_bank_details.logger_setup import setup_logging
from import_bank_details.utils import load_config

# Get the logger for this module
logger = logging.getLogger(__name__)


def get_latest_files(data_dir: str) -> Dict[str, str]:
    """
    Identify the most recently modified file in each data subfolder to process.

    Args:
        data_dir (str): Path to the data directory.

    Returns:
        dict: Dictionary with folder names as keys and latest file paths as values.
    """
    cwd = os.getcwd()
    folders_data = os.listdir(data_dir)
    file_data = {}
    for folder in folders_data:
        folder_path = os.path.join(cwd, data_dir, folder)
        files = glob.glob(os.path.join(folder_path, "*"))
        if files:
            file_data[folder] = max(files, key=os.path.getctime)
        else:
            logger.warning(f"No files found in folder: {folder}")

    if not file_data:
        logger.error("No files found in any folder")
        raise ValueError("No files found in any folder")

    logger.info("Latest files from all data folders obtained.")
    return file_data


def import_data(file_path: str, import_params: Optional[Dict] = None) -> pd.DataFrame:
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
            logger.info(f"Data imported with parameters from {file_path}.")
        else:
            df = pd.read_csv(file_path)
            logger.info(f"Data imported without parameters from {file_path}.")
    except UnicodeDecodeError:
        df = pd.read_excel(file_path)
        logger.info(f"Data imported as Excel from {file_path}.")
    except Exception as e:
        logger.error(f"Error importing data from {file_path}: {e}")
        raise
    return df


def process_data(df: pd.DataFrame, config: Dict, bank_name: str) -> pd.DataFrame:
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
    if "Bank" in config["columns_old"]:
        df["Bank"] = bank_name
        logger.info(f"Bank column set for {bank_name}.")

    # Select and rename columns as per the new configuration mapping
    df = df[config["columns_old"]]
    df = df.rename(columns=dict(zip(config["columns_old"], config["columns_new"])))
    logger.info(f"Columns selected and renamed for {bank_name}.")

    # Call remove_unnecessary_expenses if the 'Remove' key exists in config
    if "Remove" in config:
        df = remove_unnecessary_expenses(df, config["Remove"])
        logger.info(f"Unnecessary expenses removed for {bank_name}.")

    # Parse the 'Day' column into datetime format as specified in the configuration
    df["Day"] = pd.to_datetime(df["Day"], format=config["Day"])
    logger.info(f"Day column converted to datetime for {bank_name}.")

    return df


def process_examples(df_examples: pd.DataFrame) -> pd.DataFrame:
    """
    Process the df_examples DataFrame to ensure correct data types for 'Day' and 'Amount'.

    Args:
        df_examples (pd.DataFrame): DataFrame containing example expenses.

    Returns:
        pd.DataFrame: Processed DataFrame with correct data types.
    """
    # Convert 'Day' to datetime
    df_examples["Day"] = pd.to_datetime(df_examples["Day"], format="%d/%m/%Y", errors="coerce")

    # Process 'Amount' column to remove currency symbols and convert to float
    df_examples["Amount"] = (
        df_examples["Amount"]
        .astype(str)  # Ensure the data is of type string
        .str.replace("€", "", regex=False)  # Remove currency symbol
        .str.replace(" ", "", regex=False)  # Remove any spaces
        .str.replace(".", "", regex=False)  # Remove thousand separator dots
        .str.replace(",", ".", regex=False)  # Replace decimal comma with dot
        .astype(float)
    )

    # Log the processing
    logger.info("Processed 'Day' and 'Amount' columns in df_examples.")

    return df_examples


def remove_unnecessary_expenses(df: pd.DataFrame, remove_criteria: List[str]) -> pd.DataFrame:
    """
    Adjust the dataframe by removing rows based on 'remove_criteria' in 'Expense_name' column,
    replacing NaN values with an empty string to avoid TypeError with bitwise NOT operator.

    Args:
        df (pd.DataFrame): DataFrame to be filtered.
        remove_criteria (list): List of strings to be removed from the 'Expense_name' column.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    # Replace NaN values with an empty string and ensure that 'Expense_name' is a string
    df["Expense_name"] = df["Expense_name"].fillna("").astype(str)

    # Use str.contains to create a boolean mask, while setting na parameter to False
    # to handle NaN values appropriately
    mask = df["Expense_name"].str.contains("|".join(remove_criteria), na=False)

    # Apply the mask using the bitwise NOT operator to filter the dataframe
    return df[~mask]


def validate_example_structure(df: pd.DataFrame, df_examples: pd.DataFrame) -> None:
    """
    Validate that all columns in df are present in df_examples and have matching data types.

    Args:
        df (pd.DataFrame): The main DataFrame.
        df_examples (pd.DataFrame): The example DataFrame to validate against.

    Raises:
        ValueError: If there are missing columns or data type mismatches.
    """
    # Check for missing columns
    missing_cols = set(df.columns) - set(df_examples.columns)
    if missing_cols:
        missing_cols_str = ", ".join(sorted(missing_cols))
        raise ValueError(
            f"Example file structure does not match data.\n"
            f"Missing Columns in Example File: {missing_cols_str}\n"
            f"Expected Columns: {', '.join(sorted(df.columns))}\n"
            f"Please ensure the example file includes all required columns."
        )

    # Check for data type mismatches
    dtype_mismatch = {}
    for col in df.columns:
        if df[col].dtype != df_examples[col].dtype:
            dtype_mismatch[col] = {"data_dtype": df[col].dtype, "example_dtype": df_examples[col].dtype}

    if dtype_mismatch:
        mismatch_details = ", ".join(
            [
                f"'{col}': data dtype is {details['data_dtype']}, " f"but example dtype is {details['example_dtype']}"
                for col, details in dtype_mismatch.items()
            ]
        )
        raise ValueError(
            f"Example file column dtypes do not match data.\n"
            f"Mismatched Columns: {mismatch_details}\n"
            f"Please ensure that the example file has the correct data types for each column."
        )

    logger.info("Example file structure and data types are valid.")


def save_to_excel(df: pd.DataFrame, output_dir: str, folders_data: List[str]) -> None:
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
    df["Day"] = df["Day"].dt.strftime("%d/%m/%Y")

    # Write the processed data to an Excel file in the output directory
    df.to_excel(os.path.join(output_dir, filename), index=False)
    logger.info(f"Dataframe saved to Excel file {filename}.")


def main() -> None:
    """Main function to orchestrate the data import, processing, and export."""
    # Set up logging
    setup_logging()

    # Load the configuration from the YAML file
    config = load_config(config_path="config_bank.yaml")

    # Identify the most recently modified file in each data subfolder to process
    file_data = get_latest_files(data_dir="data")

    # Assign the latest example file to a variable and remove it from file_data
    latest_example_file = file_data.pop("examples", None)

    # Initialize the dataframe that will hold all the data
    df = None

    # Iterate through each file, process and clean the data according to the configuration
    for bank_name, file_path in file_data.items():
        try:
            # Import data with specified parameters if any, otherwise default to a basic read
            df_temp = import_data(
                file_path=file_path,
                import_params=config[bank_name].get("import"),
            )

            # Process and clean the data
            df_temp = process_data(df=df_temp, config=config[bank_name], bank_name=bank_name)

            # Combine the current file's data with the main dataframe
            df = pd.concat([df, df_temp], ignore_index=True)
        except Exception as e:
            logger.error(f"Error processing data for {bank_name}: {e}")
            continue

    if df is not None:
        # Round down the 'Day' column to the nearest day
        df["Day"] = df["Day"].dt.floor("D")

        # Sort the combined data for uniformity and easier analysis
        df = df.sort_values(by=["Day", "Expense_name", "Amount"])

        # Convert the 'Amount' to a negative value to indicate expense
        df["Amount"] = -df["Amount"]

        # Load example expenses to help the classifier if available
        if latest_example_file is not None:
            logger.info(f"Loading example file: {latest_example_file}")
            df_examples = import_data(file_path=latest_example_file)

            # Process df_examples to ensure correct data types
            df_examples = process_examples(df_examples=df_examples)

            try:
                # Validate the example file structure and data types
                validate_example_structure(df=df, df_examples=df_examples)
            except ValueError as ve:
                logger.error(f"Validation failed: {ve}")
                raise
        else:
            logger.warning("No example file found. Classification may be less accurate.")
            df_examples = pd.DataFrame(columns=df.columns)

        # Classify the expenses with OpenAI
        logger.info("Classifying expenses with OpenAI.")
        df = classify_expenses(df=df, df_examples=df_examples, include_categories_in_prompt=True, include_online_search=True)
        logger.info("Classification complete.")

        # Save the processed data to an Excel file in the output directory
        logger.info("Saving the processed data to an Excel file.")
        save_to_excel(df, "output", list(file_data.keys()))


if __name__ == "__main__":
    main()
