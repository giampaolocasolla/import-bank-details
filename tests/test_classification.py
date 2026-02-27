"""Tests for the classification module."""

import threading
from unittest import mock
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from import_bank_details.classification import (
    classify_expenses,
    create_nested_category_string,
    get_classification,
    get_list_expenses,
)
from import_bank_details.structured_output import ExpenseEntry, ExpenseOutput, ExpenseType


class MockParsedResponse:
    """Mock for the parsed response from OpenAI Responses API."""

    def __init__(self, expense_type):
        self.output_parsed = expense_type


def test_get_list_expenses(sample_processed_df):
    """Test the get_list_expenses function."""
    # Add Primary and Secondary columns for testing with include_output=True
    df_with_categories = sample_processed_df.copy()
    df_with_categories["Primary"] = ["Groceries", "Out", "Transport", "Out"]
    df_with_categories["Secondary"] = ["Auchan", "Restaurants", "Taxi", "Bar"]

    # Call the function with include_output=True
    expenses = get_list_expenses(df=df_with_categories, include_output=True)

    # Check if the expenses list was created correctly
    assert len(expenses) == 4
    assert all(isinstance(expense, ExpenseEntry) for expense in expenses)

    # Check attributes
    assert expenses[0].input.Expense_name == "Supermarket"
    assert expenses[0].input.Amount == "-45.50"

    # Check if output was included
    assert expenses[0].output is not None

    # Call the function with include_output=False
    expenses = get_list_expenses(df=sample_processed_df, include_output=False)

    # Check if outputs are None
    assert all(expense.output is None for expense in expenses)


def test_create_nested_category_string():
    """Test the create_nested_category_string function."""
    # Test with ExpenseOutput model
    categories_str = create_nested_category_string(ExpenseOutput)

    # Check if the categories string was created correctly
    assert "Here is the nested list of Primary and Secondary categories for my expenses:" in categories_str
    assert "- Housing" in categories_str
    assert "    - Rent" in categories_str
    assert "- Transport" in categories_str
    assert "    - Fuel" in categories_str


def test_get_classification():
    """Test the get_classification function."""
    # Set up mock for OpenAI client
    mock_openai_client = MagicMock()

    # Define the expense input
    expense_input = {"Day": "01/01/2023", "Expense_name": "Supermarket", "Amount": "45.50", "Bank": "N26", "Comment": "Groceries"}

    # Define the examples
    examples = [
        {
            "input": {"Day": "02/01/2023", "Expense_name": "Lidl", "Amount": "30.25", "Bank": "Revolut", "Comment": ""},
            "output": "Groceries, Lidl",
        }
    ]

    # Create a mock expense output
    expense_type = None
    for et in ExpenseType:  # type: ignore[attr-defined]
        if et.value == "Groceries, Auchan":
            expense_type = et
            break

    mock_output = ExpenseOutput(expense_type=expense_type)

    # Mock the OpenAI API call
    mock_openai_client.responses.parse.return_value = MockParsedResponse(mock_output)

    # Call the function
    response = get_classification(
        expense_input=expense_input,
        openai_client=mock_openai_client,
        examples=examples,
        system_prompt="Test prompt",
        model_name="gpt-4o-mini",
        temperature=0.0,
        response_format=ExpenseOutput,
        include_categories_in_prompt=True,
        include_online_search=False,
    )

    # Check if the function returned the expected result
    assert isinstance(response, ExpenseOutput)
    assert response.category == "Groceries"
    assert response.subcategory == "Auchan"

    # Check if the parse method was called correctly
    mock_openai_client.responses.parse.assert_called_once()
    args, kwargs = mock_openai_client.responses.parse.call_args

    # Check the model name
    assert kwargs["model"] == "gpt-4o-mini"

    # Check if the temperature was set correctly
    assert kwargs["temperature"] == 0.0

    # Check that system prompt is passed as instructions
    assert kwargs["instructions"] == "Test prompt\n\n" + create_nested_category_string(ExpenseOutput)

    # Check if the example was included in the input messages (no system message in list)
    assert len(kwargs["input"]) > 1
    assert kwargs["input"][0]["role"] == "user"
    assert (
        kwargs["input"][0]["content"]
        == '{"Day": "02/01/2023", "Expense_name": "Lidl", "Amount": "30.25", "Bank": "Revolut", "Comment": ""}'
    )


@mock.patch("import_bank_details.classification.get_classification")
def test_classify_expenses(mock_get_classification):
    """Test the classify_expenses function."""
    # Create a sample dataframe
    df = pd.DataFrame(
        {
            "Day": pd.to_datetime(["2023-01-01", "2023-01-02"]),
            "Expense_name": ["Supermarket", "Restaurant"],
            "Amount": [45.50, 26.75],
            "Bank": ["N26", "N26"],
            "Comment": ["Groceries", "Dinner"],
        }
    )

    # Create an example dataframe
    df_examples = pd.DataFrame(
        {
            "Day": pd.to_datetime(["2023-01-03"]),
            "Expense_name": ["Lidl"],
            "Amount": [35.50],
            "Bank": ["Revolut"],
            "Comment": [""],
            "Primary": ["Groceries"],
            "Secondary": ["Lidl"],
        }
    )

    # Find the actual ExpenseType values
    expense_type_groceries = None
    expense_type_restaurants = None

    for et in ExpenseType:  # type: ignore[attr-defined]
        if et.value == "Groceries, Auchan":
            expense_type_groceries = et
        elif et.value == "Out, Restaurants":
            expense_type_restaurants = et

    # Create mock outputs
    mock_output1 = ExpenseOutput(expense_type=expense_type_groceries)
    mock_output2 = ExpenseOutput(expense_type=expense_type_restaurants)

    # Thread-safe mock side_effect
    lock = threading.Lock()

    def side_effect(**kwargs):
        with lock:
            expense_name = kwargs["expense_input"]["Expense_name"]
            if expense_name == "Supermarket":
                return mock_output1
            return mock_output2

    mock_get_classification.side_effect = side_effect

    mock_openai_client = MagicMock()

    # Call classify_expenses
    result_df = classify_expenses(
        df=df,
        df_examples=df_examples,
        openai_client=mock_openai_client,
        system_prompt="Test prompt",
        model_name="gpt-4o-mini",
        temperature=0.0,
        response_format=ExpenseOutput,
        include_categories_in_prompt=True,
        include_online_search=False,
    )

    # Check the result
    assert "Primary" in result_df.columns
    assert "Secondary" in result_df.columns

    # Sort by Expense_name to ensure consistent order
    result_df = result_df.sort_values(by="Expense_name").reset_index()

    assert result_df.loc[0, "Primary"] == "Out"
    assert result_df.loc[0, "Secondary"] == "Restaurants"
    assert result_df.loc[1, "Primary"] == "Groceries"
    assert result_df.loc[1, "Secondary"] == "Auchan"

    # Check if get_classification was called twice
    assert mock_get_classification.call_count == 2


@mock.patch("import_bank_details.classification.get_classification")
def test_classify_expenses_skip_negative(mock_get_classification):
    """Test the classify_expenses function skips expenses with negative amount."""
    # Create a sample dataframe with negative amount
    df = pd.DataFrame(
        {
            "Day": pd.to_datetime(["2023-01-01", "2023-01-02"]),
            "Expense_name": ["Supermarket", "Refund"],
            "Amount": [45.50, -26.75],
            "Bank": ["N26", "N26"],
            "Comment": ["Groceries", "Product return"],
        }
    )

    # Create an example dataframe
    df_examples = pd.DataFrame(
        {
            "Day": pd.to_datetime(["2023-01-03"]),
            "Expense_name": ["Lidl"],
            "Amount": [35.50],
            "Bank": ["Revolut"],
            "Comment": [""],
            "Primary": ["Groceries"],
            "Secondary": ["Lidl"],
        }
    )

    # Find ExpenseType for Groceries, Auchan
    expense_type_groceries = None
    for et in ExpenseType:  # type: ignore[attr-defined]
        if et.value == "Groceries, Auchan":
            expense_type_groceries = et
            break

    # Create mock output
    mock_output = ExpenseOutput(expense_type=expense_type_groceries)

    # Mock the get_classification function
    mock_get_classification.return_value = mock_output

    mock_openai_client = MagicMock()

    # Call classify_expenses
    result_df = classify_expenses(
        df=df,
        df_examples=df_examples,
        openai_client=mock_openai_client,
        system_prompt="Test prompt",
        model_name="gpt-4o-mini",
        temperature=0.0,
        response_format=ExpenseOutput,
        include_categories_in_prompt=True,
        include_online_search=False,
    )

    # Check the result
    assert "Primary" in result_df.columns
    assert "Secondary" in result_df.columns

    # The first row should be classified
    assert result_df["Primary"].iloc[0] == "Groceries"
    assert result_df["Secondary"].iloc[0] == "Auchan"

    # The second row (negative amount) should not be classified
    assert pd.isna(result_df["Primary"].iloc[1])
    assert pd.isna(result_df["Secondary"].iloc[1])

    # Check if get_classification was called only once (for the positive amount)
    assert mock_get_classification.call_count == 1


@patch("time.sleep", return_value=None)
def test_get_classification_retries(mock_sleep):
    """Test that get_classification retries on transient failure then succeeds."""
    mock_openai_client = MagicMock()

    expense_type = None
    for et in ExpenseType:  # type: ignore[attr-defined]
        if et.value == "Groceries, Auchan":
            expense_type = et
            break

    mock_output = ExpenseOutput(expense_type=expense_type)

    # First call fails, second succeeds
    mock_openai_client.responses.parse.side_effect = [
        Exception("Temporary error"),
        MockParsedResponse(mock_output),
    ]

    expense_input = {"Day": "01/01/2023", "Expense_name": "Lidl", "Amount": "30.00", "Bank": "N26", "Comment": ""}

    result = get_classification(
        expense_input=expense_input,
        openai_client=mock_openai_client,
        system_prompt="Test",
        model_name="gpt-4o-mini",
        temperature=0.0,
    )

    assert isinstance(result, ExpenseOutput)
    assert result.category == "Groceries"
    assert mock_openai_client.responses.parse.call_count == 2
    mock_sleep.assert_called_once()


@patch("time.sleep", return_value=None)
def test_get_classification_retries_exhausted(mock_sleep):
    """Test that get_classification raises after all retries are exhausted."""
    mock_openai_client = MagicMock()
    mock_openai_client.responses.parse.side_effect = Exception("Persistent error")

    expense_input = {"Day": "01/01/2023", "Expense_name": "Lidl", "Amount": "30.00", "Bank": "N26", "Comment": ""}

    with pytest.raises(Exception, match="Persistent error"):
        get_classification(
            expense_input=expense_input,
            openai_client=mock_openai_client,
            system_prompt="Test",
            model_name="gpt-4o-mini",
            temperature=0.0,
        )

    assert mock_openai_client.responses.parse.call_count == 3
    assert mock_sleep.call_count == 2
