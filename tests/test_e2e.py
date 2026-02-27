"""End-to-end tests for the full pipeline with mocked API calls only."""

import os
import threading
from unittest import mock

import pandas as pd
import pytest
import yaml

from import_bank_details.main import main
from import_bank_details.structured_output import ExpenseOutput, ExpenseType


@pytest.fixture
def e2e_data_dir(tmp_path):
    """Create a complete data directory structure for E2E testing."""
    data_dir = tmp_path / "data"
    for folder in ["n26", "revolut", "examples"]:
        (data_dir / folder).mkdir(parents=True)

    # Create N26 CSV - amounts are positive from bank exports, but main()
    # negates them. _classify_single_expense skips negative amounts, so we
    # use negative values here so they become positive after negation.
    n26_df = pd.DataFrame(
        {
            "Value Date": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "Partner Name": ["Lidl", "Shell", "Auchan"],
            "Amount (EUR)": [-45.50, -60.00, -30.25],
            "Bank": ["N26", "N26", "N26"],
            "Payment Reference": ["Weekly groceries", "Fuel", "Groceries"],
        }
    )
    n26_df.to_csv(data_dir / "n26" / "n26_jan.csv", index=False)

    # Create Revolut CSV
    revolut_df = pd.DataFrame(
        {
            "Started Date": ["2023-01-02 12:30:00", "2023-01-04 09:00:00"],
            "Description": ["Uber", "Starbucks"],
            "Amount": [-12.50, -5.75],
            "Bank": ["Revolut", "Revolut"],
            "Type": ["Transport", "Food"],
        }
    )
    revolut_df.to_csv(data_dir / "revolut" / "revolut_jan.csv", index=False)

    # Create examples CSV
    examples_df = pd.DataFrame(
        {
            "Day": ["01/01/2023", "02/01/2023"],
            "Expense_name": ["Rewe", "BVG"],
            "Amount": ["€25,00", "€3,50"],
            "Bank": ["N26", "Revolut"],
            "Comment": ["Groceries", "Public transport"],
            "Primary": ["Groceries", "Transport"],
            "Secondary": ["Rewe", "PublicTransport"],
        }
    )
    examples_df.to_csv(data_dir / "examples" / "examples.csv", index=False)

    # Create config_bank.yaml
    bank_config = {
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
            "Remove": ["To EUR"],
            "Day": "%Y-%m-%d %H:%M:%S",
        },
    }
    with open(tmp_path / "config_bank.yaml", "w") as f:
        yaml.dump(bank_config, f)

    # Create config_llm.yaml
    llm_config = {
        "llm": {"model_name": "gpt-4o-mini", "temperature_base": 0.0},
        "system_prompt": "You are a helpful assistant.",
    }
    with open(tmp_path / "config_llm.yaml", "w") as f:
        yaml.dump(llm_config, f)

    # Create output dir
    (tmp_path / "output").mkdir()

    return tmp_path


@pytest.mark.e2e
def test_full_pipeline(e2e_data_dir):
    """Full pipeline test: CSV import -> processing -> classification -> Excel output.

    Only OpenAI and Tavily API calls are mocked.
    """
    # Build classification lookup
    expense_types = {}
    for et in ExpenseType:  # type: ignore[attr-defined]
        if et.value == "Groceries, Lidl":
            expense_types["Lidl"] = et
        elif et.value == "Transport, Fuel":
            expense_types["Shell"] = et
        elif et.value == "Groceries, Auchan":
            expense_types["Auchan"] = et
        elif et.value == "Transport, Taxi":
            expense_types["Uber"] = et
        elif et.value == "Out, Bar":
            expense_types["Starbucks"] = et

    default_type = None
    for et in ExpenseType:  # type: ignore[attr-defined]
        if et.value == "OtherExpenses, OtherExpenses":
            default_type = et
            break

    lock = threading.Lock()

    def mock_classify_func(**kwargs):
        with lock:
            name = kwargs["expense_input"]["Expense_name"]
            for key, etype in expense_types.items():
                if key in name:
                    return ExpenseOutput(expense_type=etype)
            return ExpenseOutput(expense_type=default_type)

    original_cwd = os.getcwd()
    os.chdir(e2e_data_dir)

    try:
        with (
            mock.patch("import_bank_details.classification.get_classification") as mock_classify,
            mock.patch("import_bank_details.main.load_dotenv"),
            mock.patch("import_bank_details.main.OpenAI"),
            mock.patch("import_bank_details.main.TavilyClient"),
            mock.patch("import_bank_details.main.SearchCache"),
        ):

            mock_classify.side_effect = mock_classify_func

            main()

        # Verify output
        output_files = os.listdir(e2e_data_dir / "output")
        assert len(output_files) == 1
        assert output_files[0].endswith(".xlsx")

        df_result = pd.read_excel(e2e_data_dir / "output" / output_files[0])

        # Check correct columns exist
        assert "Day" in df_result.columns
        assert "Expense_name" in df_result.columns
        assert "Amount" in df_result.columns
        assert "Bank" in df_result.columns
        assert "Primary" in df_result.columns
        assert "Secondary" in df_result.columns

        # Check correct row count (3 from N26 + 2 from Revolut = 5)
        assert len(df_result) == 5

        # Check date format (DD/MM/YYYY)
        assert "/" in df_result["Day"].iloc[0]

        # Check all rows have classification
        assert df_result["Primary"].notna().all()
        assert df_result["Secondary"].notna().all()

    finally:
        os.chdir(original_cwd)
        # Clean up log handlers
        import logging

        logger = logging.getLogger()
        for handler in logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                handler.close()
                logger.removeHandler(handler)
