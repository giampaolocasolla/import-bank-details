"""Common fixtures for all tests."""

import os
import tempfile

import pandas as pd
import pytest
import yaml

from import_bank_details.structured_output import ExpenseEntry, ExpenseInput, ExpenseOutput, ExpenseType


@pytest.fixture
def sample_data_dir():
    """Create a temporary directory structure mimicking the data directory."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create subdirectories
        folders = ["n26", "revolut", "examples"]
        for folder in folders:
            os.makedirs(os.path.join(tmp_dir, folder), exist_ok=True)

        yield tmp_dir


@pytest.fixture
def sample_config():
    """Create a sample configuration dict."""
    return {
        "n26": {
            "import": {},
            "columns_old": ["Value Date", "Partner Name", "Amount (EUR)", "Bank", "Payment Reference"],
            "columns_new": ["Day", "Expense_name", "Amount", "Bank", "Comment"],
            "Day": "%Y-%m-%d",
        },
        "revolut": {
            "import": {},
            "columns_old": ["Started Date", "Description", "Amount", "Bank", "Type"],
            "columns_new": ["Day", "Expense_name", "Amount", "Bank", "Comment"],
            "Remove": ["To EUR", "Payment from Giampaolo Casolla"],
            "Day": "%Y-%m-%d %H:%M:%S",
        },
    }


@pytest.fixture
def sample_config_file(sample_config, sample_data_dir):
    """Create a sample configuration file."""
    config_path = os.path.join(sample_data_dir, "test_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(sample_config, f)
    yield config_path
    # Cleanup happens automatically due to the temporary directory


@pytest.fixture
def sample_n26_df():
    """Create a sample n26 bank dataframe."""
    data = {
        "Value Date": ["2023-01-01", "2023-01-02"],
        "Partner Name": ["Supermarket", "Restaurant"],
        "Amount (EUR)": [45.50, 26.75],
        "Bank": ["N26", "N26"],
        "Payment Reference": ["Groceries", "Dinner"],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_revolut_df():
    """Create a sample revolut bank dataframe."""
    data = {
        "Started Date": ["2023-01-03 12:34:56", "2023-01-04 09:45:23"],
        "Description": ["Transport", "Coffee Shop"],
        "Amount": [12.50, 3.75],
        "Bank": ["Revolut", "Revolut"],
        "Type": ["Transport", "Food"],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_example_df():
    """Create a sample example dataframe with classifications."""
    data = {
        "Day": ["01/01/2023", "02/01/2023", "03/01/2023"],
        "Expense_name": ["Lidl", "Uber", "Amazon"],
        "Amount": ["45.50", "12.20", "35.99"],
        "Bank": ["N26", "Revolut", "Curve"],
        "Comment": ["", "", "Books"],
        "Primary": ["Groceries", "Transport", "Leisure"],
        "Secondary": ["Lidl", "Taxi", "Books"],
    }
    df = pd.DataFrame(data)
    df["Day"] = pd.to_datetime(df["Day"], format="%d/%m/%Y")
    return df


@pytest.fixture
def sample_processed_df():
    """Create a sample processed dataframe."""
    data = {
        "Day": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"]),
        "Expense_name": ["Supermarket", "Restaurant", "Transport", "Coffee Shop"],
        "Amount": [-45.50, -26.75, -12.50, -3.75],
        "Bank": ["N26", "N26", "Revolut", "Revolut"],
        "Comment": ["Groceries", "Dinner", "Transport", "Food"],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_n26_csv(sample_n26_df, sample_data_dir):
    """Create a sample n26 CSV file."""
    file_path = os.path.join(sample_data_dir, "n26", "n26_test.csv")
    sample_n26_df.to_csv(file_path, index=False)
    return file_path


@pytest.fixture
def sample_revolut_csv(sample_revolut_df, sample_data_dir):
    """Create a sample revolut CSV file."""
    file_path = os.path.join(sample_data_dir, "revolut", "revolut_test.csv")
    sample_revolut_df.to_csv(file_path, index=False)
    return file_path


@pytest.fixture
def sample_example_csv(sample_example_df, sample_data_dir):
    """Create a sample example CSV file."""
    file_path = os.path.join(sample_data_dir, "examples", "examples_test.csv")
    # Convert datetime to string for saving to CSV
    df_to_save = sample_example_df.copy()
    df_to_save["Day"] = df_to_save["Day"].dt.strftime("%d/%m/%Y")
    df_to_save.to_csv(file_path, index=False)
    return file_path


@pytest.fixture
def mock_expense_entry():
    """Create a mock expense entry."""
    expense_input = ExpenseInput(Day="01/01/2023", Expense_name="Supermarket", Amount="45.50", Bank="N26", Comment="Groceries")

    expense_type = None
    for et in ExpenseType:
        if et.value == "Groceries, Auchan":
            expense_type = et
            break

    expense_output = None
    if expense_type:
        expense_output = ExpenseOutput(expense_type=expense_type)

    return ExpenseEntry(input=expense_input, output=expense_output)
