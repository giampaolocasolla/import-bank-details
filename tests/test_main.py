"""Tests for the main module functions."""

import os
from unittest import mock

import pandas as pd
import pytest

from import_bank_details.main import (
    detect_bank_config,
    get_latest_files,
    import_data,
    process_data,
    process_examples,
    remove_unnecessary_expenses,
    save_to_excel,
    validate_example_structure,
)


@pytest.mark.skip(reason="Path issues with mock directories")
def test_get_latest_files(sample_data_dir, sample_n26_csv, sample_revolut_csv, sample_example_csv):
    """Test the get_latest_files function."""
    # Call the function
    with mock.patch("os.getcwd", return_value=os.path.dirname(sample_data_dir)):
        file_data = get_latest_files(data_dir=os.path.basename(sample_data_dir))

    # Check if the function returned the expected files
    assert "n26" in file_data
    assert "revolut" in file_data
    assert "examples" in file_data
    assert os.path.basename(file_data["n26"]) == "n26_test.csv"
    assert os.path.basename(file_data["revolut"]) == "revolut_test.csv"
    assert os.path.basename(file_data["examples"]) == "examples_test.csv"


@pytest.mark.skip(reason="Path issues with mock directories")
def test_get_latest_files_empty_folder(sample_data_dir):
    """Test the get_latest_files function with an empty folder."""
    # Create an empty data dir
    empty_data_dir = os.path.join(sample_data_dir, "empty_data")
    os.makedirs(empty_data_dir, exist_ok=True)

    # Test with the empty directory
    with mock.patch("os.getcwd", return_value=os.path.dirname(sample_data_dir)):
        with pytest.raises(ValueError, match="No files found in any folder"):
            get_latest_files(data_dir=os.path.basename(empty_data_dir))


def test_import_data_csv(sample_n26_csv):
    """Test the import_data function with a CSV file."""
    # Import the CSV file
    df = import_data(file_path=sample_n26_csv)

    # Check if the dataframe was created correctly
    assert isinstance(df, pd.DataFrame)
    assert "Value Date" in df.columns
    assert "Partner Name" in df.columns
    assert "Amount (EUR)" in df.columns
    assert "Bank" in df.columns
    assert "Payment Reference" in df.columns
    assert df.shape[0] == 2  # Two rows in the sample data


def test_import_data_with_params(sample_revolut_csv):
    """Test the import_data function with parameters."""
    # Import with specific parameters
    import_params = {"sep": ",", "header": 0}
    df = import_data(file_path=sample_revolut_csv, import_params=import_params)

    # Check if the dataframe was created correctly
    assert isinstance(df, pd.DataFrame)
    assert "Started Date" in df.columns
    assert "Description" in df.columns
    assert "Amount" in df.columns
    assert "Bank" in df.columns
    assert "Type" in df.columns
    assert df.shape[0] == 2  # Two rows in the sample data


def test_import_data_error():
    """Test the import_data function raises an exception for non-existent file."""
    with pytest.raises(Exception):
        import_data(file_path="non_existent_file.csv")


def test_process_data(sample_n26_df, sample_config):
    """Test the process_data function."""
    # Process the dataframe
    df_processed = process_data(df=sample_n26_df, config=sample_config["n26"], bank_name="n26")

    # Check if the columns were renamed correctly
    assert "Day" in df_processed.columns
    assert "Expense_name" in df_processed.columns
    assert "Amount" in df_processed.columns
    assert "Bank" in df_processed.columns
    assert "Comment" in df_processed.columns

    # Check if the bank name was set correctly
    assert df_processed["Bank"].iloc[0] == "n26"

    # Check if Day was converted to datetime
    assert pd.api.types.is_datetime64_dtype(df_processed["Day"])


def test_process_data_with_remove(sample_revolut_df, sample_config):
    """Test the process_data function with remove criteria."""
    # Add a row that should be removed
    sample_revolut_df = pd.concat(
        [
            sample_revolut_df,
            pd.DataFrame(
                [
                    {
                        "Started Date": "2023-01-05 10:20:30",
                        "Description": "Payment from Giampaolo Casolla",
                        "Amount": 100.00,
                        "Bank": "Revolut",
                        "Type": "Income",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    # Process the dataframe
    df_processed = process_data(df=sample_revolut_df, config=sample_config["revolut"], bank_name="revolut")

    # Check if the row was removed
    assert df_processed.shape[0] == 2  # Should still have only 2 rows after removal
    assert "Payment from Giampaolo Casolla" not in df_processed["Expense_name"].values


def test_process_examples():
    """Test the process_examples function."""
    # Create a dataframe with example data
    data = {
        "Day": ["01/01/2023", "02/01/2023"],
        "Expense_name": ["Supermarket", "Restaurant"],
        "Amount": ["€45,50", "€26,75"],
        "Bank": ["N26", "N26"],
        "Comment": ["", ""],
        "Primary": ["Groceries", "Out"],
        "Secondary": ["Auchan", "Restaurants"],
    }
    df_examples = pd.DataFrame(data)

    # Process the examples
    processed_df = process_examples(df_examples=df_examples)

    # Check if 'Day' is converted to datetime
    assert pd.api.types.is_datetime64_dtype(processed_df["Day"])

    # Check if 'Amount' is converted to float
    assert pd.api.types.is_float_dtype(processed_df["Amount"])
    assert processed_df["Amount"].iloc[0] == 45.50
    assert processed_df["Amount"].iloc[1] == 26.75


def test_remove_unnecessary_expenses():
    """Test the remove_unnecessary_expenses function."""
    # Create a dataframe
    data = {"Expense_name": ["Supermarket", "Restaurant", "Payment from User", "To EUR", None]}
    df = pd.DataFrame(data)

    # Define removal criteria
    remove_criteria = ["Payment from", "To EUR"]

    # Remove unnecessary expenses
    filtered_df = remove_unnecessary_expenses(df=df, remove_criteria=remove_criteria)

    # Check if the unnecessary expenses were removed
    assert filtered_df.shape[0] == 3  # Should be left with 3 rows
    assert "Payment from User" not in filtered_df["Expense_name"].values
    assert "To EUR" not in filtered_df["Expense_name"].values


def test_validate_example_structure_valid():
    """Test validate_example_structure with valid structure."""
    # Create two dataframes with matching structure
    df = pd.DataFrame(
        {
            "Day": pd.to_datetime(["2023-01-01", "2023-01-02"]),
            "Expense_name": ["Supermarket", "Restaurant"],
            "Amount": [45.50, 26.75],
            "Bank": ["N26", "N26"],
            "Comment": ["", ""],
        }
    )

    df_examples = pd.DataFrame(
        {
            "Day": pd.to_datetime(["2023-01-03", "2023-01-04"]),
            "Expense_name": ["Cafe", "Taxi"],
            "Amount": [5.50, 12.75],
            "Bank": ["Revolut", "Revolut"],
            "Comment": ["", ""],
            "Primary": ["Out", "Transport"],
            "Secondary": ["Bar", "Taxi"],
        }
    )

    # Should not raise an exception
    validate_example_structure(df=df, df_examples=df_examples)


def test_validate_example_structure_missing_columns():
    """Test validate_example_structure with missing columns."""
    # Create two dataframes where examples is missing a column
    df = pd.DataFrame(
        {
            "Day": pd.to_datetime(["2023-01-01", "2023-01-02"]),
            "Expense_name": ["Supermarket", "Restaurant"],
            "Amount": [45.50, 26.75],
            "Bank": ["N26", "N26"],
            "Comment": ["", ""],
        }
    )

    df_examples = pd.DataFrame(
        {
            "Day": pd.to_datetime(["2023-01-03", "2023-01-04"]),
            "Expense_name": ["Cafe", "Taxi"],
            "Amount": [5.50, 12.75],
            # Missing "Bank" column
            "Comment": ["", ""],
            "Primary": ["Out", "Transport"],
            "Secondary": ["Bar", "Taxi"],
        }
    )

    # Should raise a ValueError
    with pytest.raises(ValueError, match="Example file structure does not match data"):
        validate_example_structure(df=df, df_examples=df_examples)


def test_validate_example_structure_dtype_mismatch():
    """Test validate_example_structure with data type mismatch."""
    # Create two dataframes with a data type mismatch
    df = pd.DataFrame(
        {
            "Day": pd.to_datetime(["2023-01-01", "2023-01-02"]),
            "Expense_name": ["Supermarket", "Restaurant"],
            "Amount": [45.50, 26.75],  # Float type
            "Bank": ["N26", "N26"],
            "Comment": ["", ""],
        }
    )

    df_examples = pd.DataFrame(
        {
            "Day": pd.to_datetime(["2023-01-03", "2023-01-04"]),
            "Expense_name": ["Cafe", "Taxi"],
            "Amount": ["5.50", "12.75"],  # String type
            "Bank": ["Revolut", "Revolut"],
            "Comment": ["", ""],
            "Primary": ["Out", "Transport"],
            "Secondary": ["Bar", "Taxi"],
        }
    )

    # Should raise a ValueError
    with pytest.raises(ValueError, match="Example file column dtypes do not match data"):
        validate_example_structure(df=df, df_examples=df_examples)


def test_detect_bank_config_italian_revolut(tmpdir, sample_config):
    """Test detect_bank_config with Italian Revolut CSV format."""
    # Create a temporary CSV file with Italian headers
    italian_csv = tmpdir.join("revolut_it.csv")
    italian_csv.write("Tipo,Prodotto,Data di inizio,Data di completamento,Descrizione,Importo,Costo,Valuta,State,Saldo\n")

    # Test detection
    config_key = detect_bank_config(str(italian_csv), "revolut", sample_config)

    # Should detect Italian format
    assert config_key == "revolut_it"


def test_detect_bank_config_english_revolut(tmpdir, sample_config):
    """Test detect_bank_config with English Revolut CSV format."""
    # Create a temporary CSV file with English headers
    english_csv = tmpdir.join("revolut_en.csv")
    english_csv.write("Type,Product,Started Date,Completed Date,Description,Amount,Fee,Currency,State,Balance\n")

    # Test detection
    config_key = detect_bank_config(str(english_csv), "revolut", sample_config)

    # Should detect English format
    assert config_key == "revolut"


def test_detect_bank_config_non_revolut(tmpdir, sample_config):
    """Test detect_bank_config with non-Revolut bank (should return bank_name unchanged)."""
    # Create a temporary CSV file
    n26_csv = tmpdir.join("n26.csv")
    n26_csv.write("Value Date,Partner Name,Amount (EUR),Bank,Payment Reference\n")

    # Test detection for n26 (should just return "n26")
    config_key = detect_bank_config(str(n26_csv), "n26", sample_config)

    # Should return the bank_name unchanged
    assert config_key == "n26"


def test_detect_bank_config_error_handling(sample_config):
    """Test detect_bank_config error handling with invalid file path."""
    # Test with non-existent file
    config_key = detect_bank_config("non_existent_file.csv", "revolut", sample_config)

    # Should fall back to bank_name on error
    assert config_key == "revolut"


def test_save_to_excel(tmpdir):
    """Test the save_to_excel function."""
    # Define the output directory
    output_dir = str(tmpdir)

    # Define the folders data
    folders_data = ["n26", "revolut"]

    # Create a sample dataframe with datetime objects
    sample_df = pd.DataFrame(
        {
            "Day": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"]),
            "Expense_name": ["Supermarket", "Restaurant", "Transport", "Coffee Shop"],
            "Amount": [-45.50, -26.75, -12.50, -3.75],
            "Bank": ["N26", "N26", "Revolut", "Revolut"],
            "Comment": ["Groceries", "Dinner", "Transport", "Food"],
        }
    )

    # Save to Excel
    save_to_excel(df=sample_df, output_dir=output_dir, folders_data=folders_data)

    # Check if the file was created
    file_list = os.listdir(output_dir)
    assert len(file_list) == 1

    # Check if the filename contains the folders data
    filename = file_list[0]
    assert "n26-revolut" in filename
    assert filename.endswith(".xlsx")

    # Check if the file contains the data
    excel_path = os.path.join(output_dir, filename)
    df_read = pd.read_excel(excel_path)

    # Check if the read dataframe matches the original
    assert df_read.shape == sample_df.shape

    # Day column should be formatted as strings in the Excel file
    # Check format matches DD/MM/YYYY
    assert df_read["Day"].iloc[0] == "01/01/2023"
