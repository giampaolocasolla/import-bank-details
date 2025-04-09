"""Integration tests for the import_bank_details package."""

import os
from unittest import mock

import pytest

from import_bank_details.main import main
from import_bank_details.structured_output import ExpenseOutput, ExpenseType


@pytest.mark.integration
@mock.patch("import_bank_details.classification.get_classification")
@mock.patch("import_bank_details.main.save_to_excel")
def test_main_integration(
    mock_save_to_excel, mock_get_classification, sample_data_dir, sample_n26_csv, sample_revolut_csv, sample_example_csv
):
    """Test the main function with real data files."""
    # Create output directory
    output_dir = os.path.join(sample_data_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    # Create a mock config file
    config_path = os.path.join(sample_data_dir, "config_bank.yaml")
    with open(config_path, "w") as f:
        f.write(
            """
n26:
  import: {}
  columns_old:
    - Value Date
    - Partner Name
    - Amount (EUR)
    - Bank
    - Payment Reference
  columns_new:
    - Day
    - Expense_name
    - Amount
    - Bank
    - Comment
  Day: "%Y-%m-%d"
revolut:
  import: {}
  columns_old:
    - Started Date
    - Description
    - Amount
    - Bank
    - Type
  columns_new:
    - Day
    - Expense_name
    - Amount
    - Bank
    - Comment
  Remove:
    - To EUR
    - Payment from Giampaolo Casolla
  Day: "%Y-%m-%d %H:%M:%S"
        """
        )

    # Create a mock config_llm file
    config_llm_path = os.path.join(sample_data_dir, "config_llm.yaml")
    with open(config_llm_path, "w") as f:
        f.write(
            """
llm:
  model_name: "gpt-4o-mini"
  temperature_base: 0.0
  temperature_retry: 0.7
  timeout: 180

system_prompt: "You are an helpful assistant that classifies expenses into categories and subcategories."
        """
        )

    # Find the actual ExpenseType for Groceries, Auchan
    expense_type_groceries = None
    expense_type_restaurants = None
    expense_type_transport = None
    expense_type_coffee = None

    for et in ExpenseType:
        if et.value == "Groceries, Auchan":
            expense_type_groceries = et
        elif et.value == "Out, Restaurants":
            expense_type_restaurants = et
        elif et.value == "Transport, Taxi":
            expense_type_transport = et
        elif et.value == "Out, Bar":
            expense_type_coffee = et

    # Create mock outputs based on expense name
    def mock_get_classification_func(**kwargs):
        expense_name = kwargs["expense_input"]["Expense_name"]
        if "Supermarket" in expense_name:
            return ExpenseOutput(expense_type=expense_type_groceries)
        elif "Restaurant" in expense_name:
            return ExpenseOutput(expense_type=expense_type_restaurants)
        elif "Transport" in expense_name:
            return ExpenseOutput(expense_type=expense_type_transport)
        elif "Coffee" in expense_name:
            return ExpenseOutput(expense_type=expense_type_coffee)
        else:
            return ExpenseOutput(expense_type=expense_type_groceries)  # Default

    # Set up the mock
    mock_get_classification.side_effect = mock_get_classification_func

    # Mock save_to_excel to avoid file creation issues
    mock_save_to_excel.return_value = None

    # Mock os.getcwd to return the sample_data_dir
    with mock.patch("os.getcwd", return_value=sample_data_dir):
        # Mock load_config to use our test configs
        with mock.patch("import_bank_details.main.load_config") as mock_load_config:
            # Mock the config methods to return our test configs
            with open(config_path) as f:
                import yaml

                bank_config = yaml.safe_load(f)
                mock_load_config.return_value = bank_config

            # Mock the setup_logging function to avoid creating log files
            with mock.patch("import_bank_details.main.setup_logging"):
                # Mock get_latest_files to return our test files
                with mock.patch("import_bank_details.main.get_latest_files") as mock_get_latest_files:
                    mock_get_latest_files.return_value = {
                        "n26": sample_n26_csv,
                        "revolut": sample_revolut_csv,
                        "examples": sample_example_csv,
                    }

                    # Run the main function
                    main()

    # Check that save_to_excel was called
    mock_save_to_excel.assert_called_once()


@pytest.mark.integration
@mock.patch("import_bank_details.classification.get_classification")
@mock.patch("import_bank_details.main.save_to_excel")
def test_main_missing_example(mock_save_to_excel, mock_get_classification, sample_data_dir, sample_n26_csv, sample_revolut_csv):
    """Test the main function when no example file is available."""
    # Create output directory
    output_dir = os.path.join(sample_data_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    # Create a mock config file
    config_path = os.path.join(sample_data_dir, "config_bank.yaml")
    with open(config_path, "w") as f:
        f.write(
            """
n26:
  import: {}
  columns_old:
    - Value Date
    - Partner Name
    - Amount (EUR)
    - Bank
    - Payment Reference
  columns_new:
    - Day
    - Expense_name
    - Amount
    - Bank
    - Comment
  Day: "%Y-%m-%d"
revolut:
  import: {}
  columns_old:
    - Started Date
    - Description
    - Amount
    - Bank
    - Type
  columns_new:
    - Day
    - Expense_name
    - Amount
    - Bank
    - Comment
  Remove:
    - To EUR
    - Payment from Giampaolo Casolla
  Day: "%Y-%m-%d %H:%M:%S"
        """
        )

    # Create a mock config_llm file
    config_llm_path = os.path.join(sample_data_dir, "config_llm.yaml")
    with open(config_llm_path, "w") as f:
        f.write(
            """
llm:
  model_name: "gpt-4o-mini"
  temperature_base: 0.0
  temperature_retry: 0.7
  timeout: 180

system_prompt: "You are an helpful assistant that classifies expenses into categories and subcategories."
        """
        )

    # Create a mock expense output
    expense_type = None
    for et in ExpenseType:
        if et.value == "Groceries, Auchan":
            expense_type = et
            break

    mock_output = ExpenseOutput(expense_type=expense_type)

    # Set up the mock
    mock_get_classification.return_value = mock_output

    # Mock save_to_excel to avoid file creation issues
    mock_save_to_excel.return_value = None

    # Mock os.getcwd to return the sample_data_dir
    with mock.patch("os.getcwd", return_value=sample_data_dir):
        # Mock load_config to use our test configs
        with mock.patch("import_bank_details.main.load_config") as mock_load_config:
            # Mock the config methods to return our test configs
            with open(config_path) as f:
                import yaml

                bank_config = yaml.safe_load(f)
                mock_load_config.return_value = bank_config

            # Mock the setup_logging function to avoid creating log files
            with mock.patch("import_bank_details.main.setup_logging"):
                # Mock get_latest_files to return our test files (without examples)
                with mock.patch("import_bank_details.main.get_latest_files") as mock_get_latest_files:
                    mock_get_latest_files.return_value = {"n26": sample_n26_csv, "revolut": sample_revolut_csv}

                    # Run the main function
                    main()

    # Check that save_to_excel was called
    mock_save_to_excel.assert_called_once()
