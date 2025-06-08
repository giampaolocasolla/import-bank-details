"""Tests for the classification module."""

import time
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock, mock_open, patch

import pandas as pd

from import_bank_details.classification import (
    SearchCache,
    classify_expenses,
    create_nested_category_string,
    get_classification,
    get_list_expenses,
    perform_online_search,
)
from import_bank_details.structured_output import ExpenseEntry, ExpenseOutput, ExpenseType


class MockParsedResponse:
    """Mock for the parsed response from OpenAI."""

    def __init__(self, expense_type):
        self.choices = [mock.MagicMock(message=mock.MagicMock(parsed=expense_type))]


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


@mock.patch("import_bank_details.classification.client.beta.chat.completions.parse")
def test_get_classification(mock_parse):
    """Test the get_classification function."""
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
    # Find the actual ExpenseType for Groceries, Auchan
    expense_type = None
    for et in ExpenseType:
        if et.value == "Groceries, Auchan":
            expense_type = et
            break

    mock_output = ExpenseOutput(expense_type=expense_type)

    # Mock the OpenAI API call
    mock_parse.return_value = MockParsedResponse(mock_output)

    # Call the function
    response = get_classification(
        expense_input=expense_input,
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
    mock_parse.assert_called_once()
    args, kwargs = mock_parse.call_args

    # Check the model name
    assert kwargs["model"] == "gpt-4o-mini"

    # Check if the temperature was set correctly
    assert kwargs["temperature"] == 0.0

    # Check if the example was included in the messages
    assert len(kwargs["messages"]) > 1
    assert kwargs["messages"][1]["role"] == "user"
    assert (
        kwargs["messages"][1]["content"]
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

    # Find the actual ExpenseType for Groceries, Auchan
    expense_type_groceries = None
    expense_type_restaurants = None

    for et in ExpenseType:
        if et.value == "Groceries, Auchan":
            expense_type_groceries = et
        elif et.value == "Out, Restaurants":
            expense_type_restaurants = et

    # Create mock outputs
    mock_output1 = ExpenseOutput(expense_type=expense_type_groceries)
    mock_output2 = ExpenseOutput(expense_type=expense_type_restaurants)

    # Mock the get_classification function to return different values based on input
    mock_get_classification.side_effect = lambda **kwargs: (
        mock_output1 if kwargs["expense_input"]["Expense_name"] == "Supermarket" else mock_output2
    )

    # Call classify_expenses
    result_df = classify_expenses(
        df=df,
        df_examples=df_examples,
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
    assert result_df["Primary"].iloc[0] == "Groceries"
    assert result_df["Secondary"].iloc[0] == "Auchan"
    assert result_df["Primary"].iloc[1] == "Out"
    assert result_df["Secondary"].iloc[1] == "Restaurants"

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
            "Amount": [45.50, -26.75],  # Negative amount for refund
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
    for et in ExpenseType:
        if et.value == "Groceries, Auchan":
            expense_type_groceries = et
            break

    # Create mock output
    mock_output = ExpenseOutput(expense_type=expense_type_groceries)

    # Mock the get_classification function
    mock_get_classification.return_value = mock_output

    # Call classify_expenses
    result_df = classify_expenses(
        df=df,
        df_examples=df_examples,
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


def test_search_cache_init():
    """Test the SearchCache initialization."""
    cache = SearchCache()
    assert cache.max_retries == 3
    assert cache.initial_delay == 2.0
    assert cache.last_request_time == 0.0
    assert cache.min_request_interval == 4.0

    custom_cache = SearchCache(max_retries=5, initial_delay=2.0)
    assert custom_cache.max_retries == 5
    assert custom_cache.initial_delay == 2.0


def test_search_cache_get_cache_path(tmpdir):
    """Test the get_cache_path method."""
    cache = SearchCache()

    # Test default path
    default_path = cache.get_cache_path()
    assert str(default_path).endswith("data/examples/search_cache.json")

    # Test custom path
    custom_dir = Path(tmpdir)
    custom_path = cache.get_cache_path(custom_dir)
    assert custom_path == custom_dir / "search_cache.json"
    assert custom_dir.exists()


def test_search_cache_load_cache():
    """Test the load_cache method."""
    cache = SearchCache()

    # Test for non-existent file
    with patch("pathlib.Path.exists", return_value=False):
        result = cache.load_cache()
        assert result == {}

    # Test for existing file
    mock_json_data = '{"test_key": "test_value"}'
    with patch("pathlib.Path.exists", return_value=True):
        with patch("builtins.open", mock_open(read_data=mock_json_data)):
            result = cache.load_cache()
            assert result == {"test_key": "test_value"}

    # Test for JSON decode error
    with patch("pathlib.Path.exists", return_value=True):
        with patch("builtins.open", mock_open(read_data="invalid json")):
            with patch("import_bank_details.classification.logger.warning") as mock_warning:
                result = cache.load_cache()
                assert result == {}
                mock_warning.assert_called_once()


def test_search_cache_save_cache(tmpdir):
    """Test the save_cache method."""
    cache = SearchCache()
    test_data = {"test_key": "test_value"}

    with patch("builtins.open", mock_open()) as mock_file:
        cache.save_cache(test_data)
        mock_file.assert_called_once()


def test_search_cache_rate_limit():
    """Test the rate_limit method."""
    cache = SearchCache()
    cache.min_request_interval = 0.1  # Set a small interval for testing

    # First call should not sleep
    start_time = time.time()
    cache.rate_limit()
    elapsed = time.time() - start_time
    assert elapsed < 0.05  # Should be almost instant

    # Second immediate call should sleep
    with patch("time.sleep") as mock_sleep:
        cache.rate_limit()
        mock_sleep.assert_called_once()


@patch("import_bank_details.classification.search_cache", new_callable=MagicMock)
@patch("import_bank_details.classification.tavily_client")
def test_perform_online_search_basic(mock_tavily_client, mock_search_cache):
    """Test the perform_online_search function with basic functionality."""
    # Set up search results
    mock_results = {"results": [{"title": "Test Result", "content": "Test Content"}]}
    mock_tavily_client.search.return_value = mock_results

    # Test with valid search term
    mock_search_cache.load_cache.return_value = {}
    result = perform_online_search("test query")

    # Verify expected calls
    mock_search_cache.load_cache.assert_called_once()
    mock_search_cache.save_cache.assert_called_once()
    mock_tavily_client.search.assert_called_once_with(query="test query", search_depth="basic", max_results=2, country="germany")

    # Check that results contain the expected content
    assert "Test Result" in result
    assert "Test Content" in result


@patch("import_bank_details.classification.search_cache", new_callable=MagicMock)
def test_perform_online_search_cached(mock_search_cache):
    """Test the perform_online_search function with cached results."""
    # Set up mock for SearchCache
    mock_search_cache.load_cache.return_value = {"test query:2": "Cached result"}

    result = perform_online_search("test query")

    # Verify cache was checked but not saved (as we got a hit)
    mock_search_cache.load_cache.assert_called_once()
    mock_search_cache.save_cache.assert_not_called()

    # Check result is from cache
    assert result == "Cached result"


@patch("import_bank_details.classification.search_cache", new_callable=MagicMock)
@patch("import_bank_details.classification.tavily_client")
def test_perform_online_search_empty_results(mock_tavily_client, mock_search_cache):
    """Test the perform_online_search function with empty results."""
    # Set up mock for Tavily with empty results
    mock_tavily_client.search.return_value = {"results": []}

    mock_search_cache.load_cache.return_value = {}
    result = perform_online_search("test query")

    # Verify expected calls
    mock_search_cache.load_cache.assert_called_once()
    # Cache should NOT be saved for empty results
    mock_search_cache.save_cache.assert_not_called()

    # Check that result is the expected "not found" message
    assert result == "No results found"


@patch("time.sleep", return_value=None)
@patch("import_bank_details.classification.search_cache", new_callable=MagicMock)
@patch("import_bank_details.classification.tavily_client")
def test_perform_online_search_retry_and_fail(mock_tavily_client, mock_search_cache, mock_sleep):
    """Test the perform_online_search function with retry logic for failures."""
    # Set up mock for Tavily to always raise an exception
    mock_tavily_client.search.side_effect = Exception("API limit exceeded")

    mock_search_cache.load_cache.return_value = {}
    # Set max_retries to a specific value for the test
    mock_search_cache.max_retries = 3
    mock_search_cache.initial_delay = 0.1

    result = perform_online_search("test query")

    # Verify it tried to load from cache
    mock_search_cache.load_cache.assert_called_once()
    # Verify it never saved to cache
    mock_search_cache.save_cache.assert_not_called()
    # Verify the number of search attempts
    assert mock_tavily_client.search.call_count == 3
    # Verify the number of sleeps
    assert mock_sleep.call_count == 3
    # Check for the final error message
    assert result == "Online search failed after multiple attempts"


@patch("time.sleep", return_value=None)
@patch("import_bank_details.classification.search_cache", new_callable=MagicMock)
@patch("import_bank_details.classification.tavily_client")
def test_perform_online_search_retry_and_succeed(mock_tavily_client, mock_search_cache, mock_sleep):
    """Test the perform_online_search function with retry logic that succeeds."""
    # Set up mock for Tavily to fail twice, then succeed
    mock_results = {"results": [{"title": "Test Result", "content": "Test Content"}]}
    mock_tavily_client.search.side_effect = [
        Exception("API limit exceeded"),
        Exception("Search error"),
        mock_results,
    ]

    mock_search_cache.load_cache.return_value = {}
    mock_search_cache.max_retries = 3
    mock_search_cache.initial_delay = 0.1

    result = perform_online_search("test query")

    # Verify it tried to load from cache
    mock_search_cache.load_cache.assert_called_once()
    # Verify it saved to cache on success
    mock_search_cache.save_cache.assert_called_once()
    # Verify the number of search attempts
    assert mock_tavily_client.search.call_count == 3
    # Verify the number of sleeps for the failed attempts
    assert mock_sleep.call_count == 2
    # Check that results contain the expected content
    assert "Test Result" in result
    assert "Test Content" in result
