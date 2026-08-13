"""Tests for the classification module."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from openai import APIStatusError, APITimeoutError
from pydantic import ValidationError

from import_bank_details.classification import (
    classify_expenses,
    create_nested_category_string,
    get_batch_classification,
    get_classification,
    get_list_expenses,
)
from import_bank_details.classification_cache import ClassificationCache
from import_bank_details.structured_output import ExpenseBatchItem, ExpenseEntry, ExpenseOutput, ExpenseOutputBatch, ExpenseType


def _timeout_error() -> APITimeoutError:
    return APITimeoutError(request=MagicMock())


def _status_error(status_code: int, message: str = "error") -> APIStatusError:
    response = MagicMock()
    response.status_code = status_code
    response.headers = {}
    return APIStatusError(message, response=response, body=None)


def _schema_error() -> ValidationError:
    return ValidationError.from_exception_data("ExpenseOutput", [{"type": "missing", "loc": ("expense_type",), "input": {}}])


def _find_expense_type(value: str):
    for et in ExpenseType:  # type: ignore[attr-defined]
        if et.value == value:
            return et
    raise AssertionError(f"ExpenseType not found: {value}")


def _mock_chat_response(parsed):
    mock_message = MagicMock()
    mock_message.parsed = parsed
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response


def _patch_cache_path(tmp_path):
    def fake_get_cache_path(self, custom_path=None):
        cache_dir = custom_path or tmp_path
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / "classification_cache.json"

    return patch.object(ClassificationCache, "get_cache_path", fake_get_cache_path)


@pytest.fixture
def isolated_cache(tmp_path):
    cache = ClassificationCache()
    with _patch_cache_path(tmp_path):
        yield cache


def _two_expense_df():
    return pd.DataFrame(
        {
            "Day": pd.to_datetime(["2023-01-01", "2023-01-02"]),
            "Expense_name": ["Supermarket", "Restaurant"],
            "Amount": [45.50, 26.75],
            "Bank": ["N26", "N26"],
            "Comment": ["Groceries", "Dinner"],
        }
    )


def _example_df():
    return pd.DataFrame(
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
        llm_client=mock_openai_client,
        examples=examples,
        system_prompt="Test prompt",
        model_name="gpt-4o-mini",
        temperature=0.0,
        response_format=ExpenseOutput,
        include_categories_in_prompt=True,
        include_online_search=False,
        provider="openai",
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
    assert "extra_body" not in kwargs
    assert "reasoning_effort" not in kwargs
    mock_openai_client.chat.completions.parse.assert_not_called()


def test_get_classification_ollama_uses_chat_completions():
    """Ollama classification should use Chat Completions with reasoning_effort=none."""
    mock_openai_client = MagicMock()

    expense_input = {"Day": "01/01/2023", "Expense_name": "Supermarket", "Amount": "45.50", "Bank": "N26", "Comment": "Groceries"}

    expense_type = None
    for et in ExpenseType:  # type: ignore[attr-defined]
        if et.value == "Groceries, Auchan":
            expense_type = et
            break

    mock_output = ExpenseOutput(expense_type=expense_type)
    mock_message = MagicMock()
    mock_message.parsed = mock_output
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_openai_client.chat.completions.parse.return_value = mock_response

    response = get_classification(
        expense_input=expense_input,
        llm_client=mock_openai_client,
        system_prompt="Test prompt",
        model_name="qwen3.5:9b-q8_0",
        temperature=0.0,
        response_format=ExpenseOutput,
        include_categories_in_prompt=False,
        reasoning_effort="none",
        provider="ollama",
    )

    assert isinstance(response, ExpenseOutput)
    assert response.category == "Groceries"
    mock_openai_client.responses.parse.assert_not_called()
    mock_openai_client.chat.completions.parse.assert_called_once()
    kwargs = mock_openai_client.chat.completions.parse.call_args.kwargs
    assert kwargs["model"] == "qwen3.5:9b-q8_0"
    assert kwargs["response_format"] is ExpenseOutput
    assert kwargs["extra_body"] == {"reasoning_effort": "none"}
    assert kwargs["messages"][0] == {"role": "system", "content": "Test prompt"}


def test_classify_expenses(isolated_cache):
    """Two expenses in one batch should produce a single parse call."""
    df = _two_expense_df()
    df_examples = _example_df()

    batch_output = ExpenseOutputBatch(
        items=[
            ExpenseBatchItem(id="0", expense_type=_find_expense_type("Groceries, Auchan")),
            ExpenseBatchItem(id="1", expense_type=_find_expense_type("Out, Restaurants")),
        ]
    )
    mock_openai_client = MagicMock()
    mock_openai_client.responses.parse.return_value = MockParsedResponse(batch_output)

    result_df = classify_expenses(
        df=df,
        df_examples=df_examples,
        llm_client=mock_openai_client,
        system_prompt="Test prompt",
        model_name="gpt-4o-mini",
        temperature=0.0,
        response_format=ExpenseOutput,
        include_categories_in_prompt=True,
        include_online_search=False,
        provider="openai",
        classification_cache=isolated_cache,
        batch_size=10,
        max_workers=1,
    )

    result_df = result_df.sort_values(by="Expense_name").reset_index()
    assert result_df.loc[0, "Primary"] == "Out"
    assert result_df.loc[0, "Secondary"] == "Restaurants"
    assert result_df.loc[1, "Primary"] == "Groceries"
    assert result_df.loc[1, "Secondary"] == "Auchan"
    assert mock_openai_client.responses.parse.call_count == 1
    mock_openai_client.chat.completions.parse.assert_not_called()
    kwargs = mock_openai_client.responses.parse.call_args.kwargs
    assert kwargs["text_format"] is ExpenseOutputBatch


def test_classify_expenses_skip_negative(isolated_cache):
    """Negative amounts skip the LLM and are not written to the cache."""
    df = pd.DataFrame(
        {
            "Day": pd.to_datetime(["2023-01-01", "2023-01-02"]),
            "Expense_name": ["Supermarket", "Refund"],
            "Amount": [45.50, -26.75],
            "Bank": ["N26", "N26"],
            "Comment": ["Groceries", "Product return"],
        }
    )
    df_examples = _example_df()

    batch_output = ExpenseOutputBatch(items=[ExpenseBatchItem(id="0", expense_type=_find_expense_type("Groceries, Auchan"))])
    mock_openai_client = MagicMock()
    mock_openai_client.responses.parse.return_value = MockParsedResponse(batch_output)

    result_df = classify_expenses(
        df=df,
        df_examples=df_examples,
        llm_client=mock_openai_client,
        system_prompt="Test prompt",
        model_name="gpt-4o-mini",
        temperature=0.0,
        response_format=ExpenseOutput,
        include_categories_in_prompt=True,
        include_online_search=False,
        provider="openai",
        classification_cache=isolated_cache,
        batch_size=10,
        max_workers=1,
    )

    assert result_df["Primary"].iloc[0] == "Groceries"
    assert result_df["Secondary"].iloc[0] == "Auchan"
    assert pd.isna(result_df["Primary"].iloc[1])
    assert pd.isna(result_df["Secondary"].iloc[1])
    assert mock_openai_client.responses.parse.call_count == 1
    assert isolated_cache.get("Supermarket") == {"Primary": "Groceries", "Secondary": "Auchan"}
    assert isolated_cache.get("Refund") is None


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

    mock_openai_client.responses.parse.side_effect = [
        _timeout_error(),
        MockParsedResponse(mock_output),
    ]

    expense_input = {"Day": "01/01/2023", "Expense_name": "Lidl", "Amount": "30.00", "Bank": "N26", "Comment": ""}

    result = get_classification(
        expense_input=expense_input,
        llm_client=mock_openai_client,
        system_prompt="Test",
        model_name="gpt-4o-mini",
        temperature=0.0,
        provider="openai",
    )

    assert isinstance(result, ExpenseOutput)
    assert result.category == "Groceries"
    assert mock_openai_client.responses.parse.call_count == 2
    mock_openai_client.chat.completions.parse.assert_not_called()
    mock_sleep.assert_called_once()


@patch("time.sleep", return_value=None)
def test_get_classification_retries_exhausted(mock_sleep):
    """Test that get_classification raises after all retries are exhausted."""
    mock_openai_client = MagicMock()
    mock_openai_client.responses.parse.side_effect = _status_error(503, "Persistent error")

    expense_input = {"Day": "01/01/2023", "Expense_name": "Lidl", "Amount": "30.00", "Bank": "N26", "Comment": ""}

    with pytest.raises(APIStatusError, match="Persistent error"):
        get_classification(
            expense_input=expense_input,
            llm_client=mock_openai_client,
            system_prompt="Test",
            model_name="gpt-4o-mini",
            temperature=0.0,
            provider="openai",
        )

    assert mock_openai_client.responses.parse.call_count == 3
    assert mock_sleep.call_count == 2


@patch("time.sleep", return_value=None)
def test_get_classification_chat_completions_retries(mock_sleep):
    """Ollama Chat Completions should retry then succeed."""
    mock_openai_client = MagicMock()

    expense_type = None
    for et in ExpenseType:  # type: ignore[attr-defined]
        if et.value == "Groceries, Auchan":
            expense_type = et
            break

    mock_output = ExpenseOutput(expense_type=expense_type)
    mock_message = MagicMock()
    mock_message.parsed = mock_output
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_openai_client.chat.completions.parse.side_effect = [
        _status_error(429, "Temporary error"),
        mock_response,
    ]

    result = get_classification(
        expense_input={"Day": "01/01/2023", "Expense_name": "Lidl", "Amount": "30.00", "Bank": "N26", "Comment": ""},
        llm_client=mock_openai_client,
        system_prompt="Test",
        model_name="qwen3.5:9b-q8_0",
        reasoning_effort="none",
        provider="ollama",
    )

    assert result.category == "Groceries"
    assert mock_openai_client.chat.completions.parse.call_count == 2
    mock_openai_client.responses.parse.assert_not_called()
    mock_sleep.assert_called_once()


@patch("time.sleep", return_value=None)
def test_get_classification_connection_error_not_retried(mock_sleep):
    """Connection failures on the Ollama path should fail immediately."""
    mock_openai_client = MagicMock()
    mock_openai_client.chat.completions.parse.side_effect = ConnectionError("Connection refused")

    with pytest.raises(ConnectionError, match="Connection refused"):
        get_classification(
            expense_input={"Day": "01/01/2023", "Expense_name": "Lidl", "Amount": "30.00", "Bank": "N26", "Comment": ""},
            llm_client=mock_openai_client,
            system_prompt="Test",
            model_name="qwen3.5:9b-q8_0",
            provider="ollama",
        )

    assert mock_openai_client.chat.completions.parse.call_count == 1
    mock_sleep.assert_not_called()


@patch("time.sleep", return_value=None)
def test_get_classification_schema_error_retries_once(mock_sleep):
    """Pydantic/schema validation errors should be retried at most once."""
    mock_openai_client = MagicMock()
    mock_openai_client.chat.completions.parse.side_effect = _schema_error()

    with pytest.raises(ValidationError):
        get_classification(
            expense_input={"Day": "01/01/2023", "Expense_name": "Lidl", "Amount": "30.00", "Bank": "N26", "Comment": ""},
            llm_client=mock_openai_client,
            system_prompt="Test",
            model_name="qwen3.5:9b-q8_0",
            provider="ollama",
        )

    assert mock_openai_client.chat.completions.parse.call_count == 2
    mock_sleep.assert_called_once()


def test_get_classification_none_parsed_raises():
    """A chat completion with no parsed payload should raise ValueError."""
    mock_openai_client = MagicMock()
    mock_message = MagicMock()
    mock_message.parsed = None
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_openai_client.chat.completions.parse.return_value = mock_response

    with pytest.raises(ValueError, match="no parsed classification"):
        get_classification(
            expense_input={"Day": "01/01/2023", "Expense_name": "Lidl", "Amount": "30.00", "Bank": "N26", "Comment": ""},
            llm_client=mock_openai_client,
            system_prompt="Test",
            model_name="qwen3.5:9b-q8_0",
            provider="ollama",
        )


def test_get_batch_classification_ollama_uses_chat_completions():
    """Ollama batch classification should use Chat Completions with ExpenseOutputBatch."""
    mock_openai_client = MagicMock()
    batch_output = ExpenseOutputBatch(items=[ExpenseBatchItem(id="0", expense_type=_find_expense_type("Groceries, Auchan"))])
    mock_openai_client.chat.completions.parse.return_value = _mock_chat_response(batch_output)

    expenses = [
        {
            "id": "0",
            "Day": "01/01/2023",
            "Expense_name": "Supermarket",
            "Amount": "45.50",
            "Bank": "N26",
            "Comment": "Groceries",
        }
    ]
    result = get_batch_classification(
        expenses=expenses,
        llm_client=mock_openai_client,
        examples=[
            {
                "input": {"Day": "02/01/2023", "Expense_name": "Lidl", "Amount": "30.25", "Bank": "Revolut", "Comment": ""},
                "output": "Groceries, Lidl",
            }
        ],
        system_prompt="Test prompt",
        model_name="qwen3.5:9b-q8_0",
        temperature=0.0,
        include_categories_in_prompt=True,
        reasoning_effort="none",
        provider="ollama",
    )

    assert isinstance(result, ExpenseOutputBatch)
    assert result.items[0].id == "0"
    mock_openai_client.responses.parse.assert_not_called()
    mock_openai_client.chat.completions.parse.assert_called_once()
    kwargs = mock_openai_client.chat.completions.parse.call_args.kwargs
    assert kwargs["response_format"] is ExpenseOutputBatch
    assert kwargs["extra_body"] == {"reasoning_effort": "none"}
    assert kwargs["messages"][0]["role"] == "system"
    assert "nested list of Primary and Secondary" in kwargs["messages"][0]["content"]
    user_contents = [message["content"] for message in kwargs["messages"] if message["role"] == "user"]
    assert any('"id": "0"' in content and "Supermarket" in content for content in user_contents)


def test_get_batch_classification_openai_uses_responses():
    """OpenAI batch classification should use the Responses API."""
    mock_openai_client = MagicMock()
    batch_output = ExpenseOutputBatch(items=[ExpenseBatchItem(id="0", expense_type=_find_expense_type("Groceries, Auchan"))])
    mock_openai_client.responses.parse.return_value = MockParsedResponse(batch_output)

    result = get_batch_classification(
        expenses=[{"id": "0", "Day": "01/01/2023", "Expense_name": "Lidl", "Amount": "30.00", "Bank": "N26", "Comment": ""}],
        llm_client=mock_openai_client,
        system_prompt="Test",
        model_name="gpt-4o-mini",
        temperature=0.0,
        provider="openai",
    )

    assert result.items[0].id == "0"
    mock_openai_client.chat.completions.parse.assert_not_called()
    kwargs = mock_openai_client.responses.parse.call_args.kwargs
    assert kwargs["text_format"] is ExpenseOutputBatch
    assert "extra_body" not in kwargs


def test_get_batch_classification_includes_search_text():
    """Tavily search text should be attached per expense in the batch payload."""
    mock_openai_client = MagicMock()
    batch_output = ExpenseOutputBatch(items=[ExpenseBatchItem(id="0", expense_type=_find_expense_type("Groceries, Auchan"))])
    mock_openai_client.chat.completions.parse.return_value = _mock_chat_response(batch_output)
    mock_tavily = MagicMock()
    mock_cache = MagicMock()

    with patch("import_bank_details.classification.perform_online_search", return_value="Lidl supermarket") as mock_search:
        get_batch_classification(
            expenses=[{"id": "0", "Day": "01/01/2023", "Expense_name": "Lidl", "Amount": "30.00", "Bank": "N26", "Comment": ""}],
            llm_client=mock_openai_client,
            system_prompt="Test",
            model_name="qwen3.5:9b-q8_0",
            include_online_search=True,
            tavily_client=mock_tavily,
            search_cache=mock_cache,
            provider="ollama",
        )

    mock_search.assert_called_once()
    user_message = mock_openai_client.chat.completions.parse.call_args.kwargs["messages"][-1]["content"]
    assert "Lidl supermarket" in user_message


def test_classify_expenses_cache_hit_skips_llm(isolated_cache):
    """A cache hit should skip the batch LLM call."""
    mock_openai_client = MagicMock()
    isolated_cache.put("Supermarket", "Groceries", "Auchan")
    result_df = classify_expenses(
        df=_two_expense_df().iloc[[0]],
        df_examples=_example_df(),
        llm_client=mock_openai_client,
        system_prompt="Test prompt",
        model_name="qwen3.5:9b-q8_0",
        provider="ollama",
        classification_cache=isolated_cache,
        batch_size=10,
        max_workers=1,
    )

    assert result_df["Primary"].iloc[0] == "Groceries"
    assert result_df["Secondary"].iloc[0] == "Auchan"
    mock_openai_client.chat.completions.parse.assert_not_called()
    mock_openai_client.responses.parse.assert_not_called()


def test_classify_expenses_writes_cache(isolated_cache):
    """A successful batch classification should be stored in the cache."""
    batch_output = ExpenseOutputBatch(items=[ExpenseBatchItem(id="0", expense_type=_find_expense_type("Groceries, Auchan"))])
    mock_openai_client = MagicMock()
    mock_openai_client.chat.completions.parse.return_value = _mock_chat_response(batch_output)

    classify_expenses(
        df=_two_expense_df().iloc[[0]],
        df_examples=_example_df(),
        llm_client=mock_openai_client,
        system_prompt="Test prompt",
        model_name="qwen3.5:9b-q8_0",
        provider="ollama",
        classification_cache=isolated_cache,
        batch_size=10,
        max_workers=1,
    )
    assert isolated_cache.get("Supermarket") == {"Primary": "Groceries", "Secondary": "Auchan"}


def test_classify_expenses_ollama_batch_is_one_parse_call(isolated_cache):
    """A batch of 2+ expenses should result in one Chat Completions parse call."""
    batch_output = ExpenseOutputBatch(
        items=[
            ExpenseBatchItem(id="0", expense_type=_find_expense_type("Groceries, Auchan")),
            ExpenseBatchItem(id="1", expense_type=_find_expense_type("Out, Restaurants")),
        ]
    )
    mock_openai_client = MagicMock()
    mock_openai_client.chat.completions.parse.return_value = _mock_chat_response(batch_output)

    result_df = classify_expenses(
        df=_two_expense_df(),
        df_examples=_example_df(),
        llm_client=mock_openai_client,
        system_prompt="Test prompt",
        model_name="qwen3.5:9b-q8_0",
        provider="ollama",
        classification_cache=isolated_cache,
        batch_size=10,
        max_workers=1,
    )

    assert mock_openai_client.chat.completions.parse.call_count == 1
    mock_openai_client.responses.parse.assert_not_called()
    result_df = result_df.sort_values(by="Expense_name").reset_index(drop=True)
    assert result_df.loc[0, "Primary"] == "Out"
    assert result_df.loc[1, "Primary"] == "Groceries"


def test_classify_expenses_missing_batch_id_falls_back_to_single(isolated_cache):
    """A missing batch id should retry that expense via get_classification."""
    batch_output = ExpenseOutputBatch(items=[ExpenseBatchItem(id="0", expense_type=_find_expense_type("Groceries, Auchan"))])
    mock_openai_client = MagicMock()
    mock_openai_client.chat.completions.parse.return_value = _mock_chat_response(batch_output)

    with patch("import_bank_details.classification.get_classification") as mock_get_classification:
        mock_get_classification.return_value = ExpenseOutput(expense_type=_find_expense_type("Out, Restaurants"))
        result_df = classify_expenses(
            df=_two_expense_df(),
            df_examples=_example_df(),
            llm_client=mock_openai_client,
            system_prompt="Test prompt",
            model_name="qwen3.5:9b-q8_0",
            provider="ollama",
            classification_cache=isolated_cache,
            batch_size=10,
            max_workers=1,
        )

    assert mock_openai_client.chat.completions.parse.call_count == 1
    assert mock_get_classification.call_count == 1
    assert mock_get_classification.call_args.kwargs["expense_input"]["Expense_name"] == "Restaurant"
    result_df = result_df.sort_values(by="Expense_name").reset_index(drop=True)
    assert result_df.loc[0, "Primary"] == "Out"
    assert result_df.loc[1, "Primary"] == "Groceries"


def test_classify_expenses_batch_size_one_still_works(isolated_cache):
    """batch_size=1 should degenerate to one expense per batch call."""
    mock_openai_client = MagicMock()
    mock_openai_client.chat.completions.parse.side_effect = [
        _mock_chat_response(
            ExpenseOutputBatch(items=[ExpenseBatchItem(id="0", expense_type=_find_expense_type("Groceries, Auchan"))])
        ),
        _mock_chat_response(
            ExpenseOutputBatch(items=[ExpenseBatchItem(id="0", expense_type=_find_expense_type("Out, Restaurants"))])
        ),
    ]

    result_df = classify_expenses(
        df=_two_expense_df(),
        df_examples=_example_df(),
        llm_client=mock_openai_client,
        system_prompt="Test prompt",
        model_name="qwen3.5:9b-q8_0",
        provider="ollama",
        classification_cache=isolated_cache,
        batch_size=1,
        max_workers=1,
    )

    assert mock_openai_client.chat.completions.parse.call_count == 2
    result_df = result_df.sort_values(by="Expense_name").reset_index(drop=True)
    assert result_df.loc[0, "Primary"] == "Out"
    assert result_df.loc[1, "Primary"] == "Groceries"


def test_classify_expenses_batch_failure_falls_back_to_single(isolated_cache):
    """If the batch LLM call fails, each expense is retried via get_classification."""
    mock_openai_client = MagicMock()
    mock_openai_client.chat.completions.parse.side_effect = ConnectionError("Connection refused")

    with patch("import_bank_details.classification.get_classification") as mock_get_classification:
        mock_get_classification.return_value = ExpenseOutput(expense_type=_find_expense_type("Groceries, Auchan"))
        result_df = classify_expenses(
            df=_two_expense_df(),
            df_examples=_example_df(),
            llm_client=mock_openai_client,
            system_prompt="Test prompt",
            model_name="qwen3.5:9b-q8_0",
            provider="ollama",
            classification_cache=isolated_cache,
            batch_size=10,
            max_workers=1,
        )

    assert mock_get_classification.call_count == 2
    assert (result_df["Primary"] == "Groceries").all()
    assert isolated_cache.get("Supermarket") == {"Primary": "Groceries", "Secondary": "Auchan"}


def test_classify_expenses_does_not_send_full_example_pool(isolated_cache):
    """A large labeled pool should be capped before the mocked LLM call."""
    n_examples = 50
    df_examples = pd.DataFrame(
        {
            "Day": pd.to_datetime(["2023-01-01"] * n_examples),
            "Expense_name": [f"Merchant{i:03d} Store" for i in range(n_examples)],
            "Amount": [10.0] * n_examples,
            "Bank": ["N26"] * n_examples,
            "Comment": [""] * n_examples,
            "Primary": ["Groceries"] * n_examples,
            "Secondary": ["Lidl"] * n_examples,
        }
    )
    df = pd.DataFrame(
        {
            "Day": pd.to_datetime(["2023-02-01"]),
            "Expense_name": ["Merchant000 Store Munich"],
            "Amount": [12.0],
            "Bank": ["N26"],
            "Comment": [""],
        }
    )
    batch_output = ExpenseOutputBatch(items=[ExpenseBatchItem(id="0", expense_type=_find_expense_type("Groceries, Lidl"))])
    mock_openai_client = MagicMock()
    mock_openai_client.chat.completions.parse.return_value = _mock_chat_response(batch_output)

    classify_expenses(
        df=df,
        df_examples=df_examples,
        llm_client=mock_openai_client,
        system_prompt="Test prompt",
        model_name="qwen3.5:9b-q8_0",
        provider="ollama",
        classification_cache=isolated_cache,
        batch_size=10,
        max_few_shot_examples=5,
        max_workers=1,
    )

    messages = mock_openai_client.chat.completions.parse.call_args.kwargs["messages"]
    example_messages = messages[1:-1]
    assert len(example_messages) == 10
    assert example_messages[0]["role"] == "user"
    assert example_messages[1]["role"] == "assistant"
