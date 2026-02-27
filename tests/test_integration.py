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
    mock_save_to_excel,
    mock_get_classification,
    sample_data_dir,
    sample_n26_csv,
    sample_revolut_csv,
    sample_example_csv,
    sample_config,
    sample_llm_config,
):
    """Test the main function with real data files."""
    os.makedirs(os.path.join(sample_data_dir, "output"), exist_ok=True)

    expense_type_groceries = None
    expense_type_restaurants = None
    expense_type_transport = None
    expense_type_coffee = None

    for et in ExpenseType:  # type: ignore[attr-defined]
        if et.value == "Groceries, Auchan":
            expense_type_groceries = et
        elif et.value == "Out, Restaurants":
            expense_type_restaurants = et
        elif et.value == "Transport, Taxi":
            expense_type_transport = et
        elif et.value == "Out, Bar":
            expense_type_coffee = et

    def mock_classify_func(**kwargs):
        expense_name = kwargs["expense_input"]["Expense_name"]
        if "Supermarket" in expense_name:
            return ExpenseOutput(expense_type=expense_type_groceries)
        elif "Restaurant" in expense_name:
            return ExpenseOutput(expense_type=expense_type_restaurants)
        elif "Transport" in expense_name:
            return ExpenseOutput(expense_type=expense_type_transport)
        elif "Coffee" in expense_name:
            return ExpenseOutput(expense_type=expense_type_coffee)
        return ExpenseOutput(expense_type=expense_type_groceries)

    mock_get_classification.side_effect = mock_classify_func
    mock_save_to_excel.return_value = None

    with (
        mock.patch("import_bank_details.main.load_config") as mock_load_config,
        mock.patch("import_bank_details.main.setup_logging"),
        mock.patch("import_bank_details.main.get_latest_files") as mock_get_latest_files,
        mock.patch("import_bank_details.main.load_dotenv"),
        mock.patch("import_bank_details.main.OpenAI"),
        mock.patch("import_bank_details.main.TavilyClient"),
        mock.patch("import_bank_details.main.SearchCache"),
    ):

        mock_load_config.side_effect = [sample_config, sample_llm_config]
        mock_get_latest_files.return_value = {
            "n26": sample_n26_csv,
            "revolut": sample_revolut_csv,
            "examples": sample_example_csv,
        }

        main()

    mock_save_to_excel.assert_called_once()


@pytest.mark.integration
@mock.patch("import_bank_details.classification.get_classification")
@mock.patch("import_bank_details.main.save_to_excel")
def test_main_missing_example(
    mock_save_to_excel,
    mock_get_classification,
    sample_data_dir,
    sample_n26_csv,
    sample_revolut_csv,
    sample_config,
    sample_llm_config,
):
    """Test the main function when no example file is available."""
    os.makedirs(os.path.join(sample_data_dir, "output"), exist_ok=True)

    expense_type = None
    for et in ExpenseType:  # type: ignore[attr-defined]
        if et.value == "Groceries, Auchan":
            expense_type = et
            break

    mock_output = ExpenseOutput(expense_type=expense_type)
    mock_get_classification.return_value = mock_output
    mock_save_to_excel.return_value = None

    with (
        mock.patch("import_bank_details.main.load_config") as mock_load_config,
        mock.patch("import_bank_details.main.setup_logging"),
        mock.patch("import_bank_details.main.get_latest_files") as mock_get_latest_files,
        mock.patch("import_bank_details.main.load_dotenv"),
        mock.patch("import_bank_details.main.OpenAI"),
        mock.patch("import_bank_details.main.TavilyClient"),
        mock.patch("import_bank_details.main.SearchCache"),
    ):

        mock_load_config.side_effect = [sample_config, sample_llm_config]
        mock_get_latest_files.return_value = {"n26": sample_n26_csv, "revolut": sample_revolut_csv}

        main()

    mock_save_to_excel.assert_called_once()
