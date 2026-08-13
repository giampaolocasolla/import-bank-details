import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, cast

import pandas as pd
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
from pydantic import BaseModel, ValidationError
from tavily import TavilyClient
from tqdm import tqdm
from tqdm.contrib.logging import tqdm_logging_redirect

from import_bank_details.classification_cache import ClassificationCache
from import_bank_details.search import SearchCache, perform_online_search
from import_bank_details.structured_output import ExpenseEntry, ExpenseInput, ExpenseOutput, ExpenseOutputBatch, ExpenseType

logger = logging.getLogger(__name__)

MAX_LLM_RETRIES = 3
MAX_SCHEMA_ATTEMPTS = 2


def _is_connection_failure(exc: BaseException) -> bool:
    """Return True when the LLM server is unreachable (do not retry)."""
    if isinstance(exc, APITimeoutError):
        return False
    return isinstance(exc, (APIConnectionError, ConnectionError))


def _is_schema_error(exc: BaseException) -> bool:
    """Return True for Pydantic/schema validation failures."""
    return isinstance(exc, ValidationError)


def _is_transient_error(exc: BaseException) -> bool:
    """Return True for timeouts, 429, 5xx, and similar retryable API errors."""
    if isinstance(exc, (APITimeoutError, TimeoutError)):
        return True
    if isinstance(exc, APIStatusError):
        status_code = getattr(exc, "status_code", None)
        return status_code == 429 or (isinstance(status_code, int) and status_code >= 500)
    return False


def _parse_structured(
    llm_client: OpenAI,
    parse_kwargs: dict[str, Any],
    use_chat_completions: bool,
) -> BaseModel:
    """Call the provider parse API with shared retry/fail-fast behavior."""
    schema_attempts = 0
    for attempt in range(MAX_LLM_RETRIES):
        try:
            if use_chat_completions:
                chat_response = llm_client.chat.completions.parse(**parse_kwargs)  # type: ignore[arg-type]
                parsed = chat_response.choices[0].message.parsed
                if parsed is None:
                    raise ValueError("Model returned no parsed classification")
                return cast(BaseModel, parsed)

            responses_result = llm_client.responses.parse(**parse_kwargs)  # type: ignore[arg-type]
            return cast(BaseModel, responses_result.output_parsed)
        except Exception as e:
            if _is_connection_failure(e):
                logger.error(f"LLM API connection failed: {e}")
                raise
            if _is_schema_error(e):
                schema_attempts += 1
                if schema_attempts >= MAX_SCHEMA_ATTEMPTS or attempt >= MAX_LLM_RETRIES - 1:
                    logger.error(f"LLM schema validation error: {e}")
                    raise
            elif not _is_transient_error(e) or attempt >= MAX_LLM_RETRIES - 1:
                logger.error(f"LLM API error after {attempt + 1} attempt(s): {e}")
                raise

            delay = 1.0 * (2**attempt)
            logger.warning(f"LLM API attempt {attempt + 1} failed: {e}. Retrying in {delay}s")
            time.sleep(delay)

    # Unreachable: the loop always returns or raises on the last iteration
    raise RuntimeError("Unexpected: retry loop exited without return or raise")


def _split_expense_type(expense_type: Any) -> tuple[str, str]:
    """Return (primary, secondary) from an ExpenseType enum member."""
    primary, secondary = expense_type.value.split(", ", 1)
    return primary, secondary


def _expense_result(expense_input: ExpenseInput, primary: str | None, secondary: str | None) -> dict[str, Any]:
    return {
        **expense_input.model_dump(),
        "Primary": primary,
        "Secondary": secondary,
    }


def _parse_amount(amount_str: str, expense_input: ExpenseInput) -> float:
    try:
        return float(amount_str)
    except ValueError:
        logger.warning(f"Invalid amount '{amount_str}' for expense: {expense_input}")
        return 0.0


def get_list_expenses(df: pd.DataFrame, include_output: bool = True) -> list[ExpenseEntry]:
    """
    Convert a DataFrame of expenses into a list of ExpenseEntry instances.

    This function processes each row of the input DataFrame and creates an ExpenseEntry,
    containing an ExpenseInput and optionally an Expense output.

    Args:
        df (pd.DataFrame): The input DataFrame containing expense data.
        include_output (bool, optional): Whether to include classification output
            in the result. Defaults to True.

    Returns:
        List[ExpenseEntry]: A list of ExpenseEntry instances.
    """
    expenses: list[ExpenseEntry] = []
    for _, row in df.iterrows():
        expense_input = ExpenseInput(
            Day=(row["Day"].strftime("%d/%m/%Y") if pd.notnull(row["Day"]) else ""),
            Expense_name=row.get("Expense_name", ""),
            Amount=(f"{row['Amount']:.2f}" if pd.notnull(row["Amount"]) else ""),
            Bank=row.get("Bank", ""),
            Comment=row.get("Comment", "") if pd.notnull(row.get("Comment")) else "",
        )
        expense_entry = ExpenseEntry(input=expense_input)

        if include_output and "Primary" in row and "Secondary" in row:
            expense_value = f"{row['Primary']}, {row['Secondary']}"
            try:
                # Attempt to create an ExpenseType enum member
                expense_type = ExpenseType(expense_value)
                expense_output = ExpenseOutput(expense_type=expense_type)
                expense_entry.output = expense_output
            except ValueError:
                # Handle the case where the expense type is invalid
                logger.error(f"Invalid expense type: {expense_value}")
                # You may choose to skip this expense or handle it differently
                expense_entry.output = None
        expenses.append(expense_entry)
    return expenses


def create_nested_category_string(response_format: type[BaseModel]) -> str:
    """
    Generates a nested list of primary and secondary categories from the BaseModel's schema.

    Args:
        response_format (Type[BaseModel]): The Pydantic BaseModel class containing the schema.

    Returns:
        str: A formatted string representing the nested categories.
    """
    schema = response_format.model_json_schema()
    enum_list = schema["$defs"]["ExpenseType"]["enum"]

    # Build a dictionary mapping primary categories to their secondary categories
    category_dict: dict[str, set[str]] = {}
    for item in enum_list:
        primary, secondary = item.split(", ")
        category_dict.setdefault(primary, set()).add(secondary)

    # Build the formatted string
    category_lines = ["Here is the nested list of Primary and Secondary categories for my expenses:\n"]
    for primary in sorted(category_dict):
        category_lines.append(f"- {primary}")
        for secondary in sorted(category_dict[primary]):
            category_lines.append(f"    - {secondary}")
    categories_str = "\n".join(category_lines)

    return categories_str


def get_classification(
    expense_input: dict[str, str],
    llm_client: OpenAI,
    examples: list[dict[str, Any]] | None = None,
    system_prompt: str = "",
    model_name: str = "gpt-5-mini",
    temperature: float | None = None,
    response_format: type[ExpenseOutput] = ExpenseOutput,
    include_categories_in_prompt: bool = False,
    include_online_search: bool = False,
    tavily_client: TavilyClient | None = None,
    search_cache: SearchCache | None = None,
    reasoning_effort: str | None = None,
    provider: str = "ollama",
) -> ExpenseOutput:
    """
    Get classification for an expense input using an OpenAI-compatible API.

    Local Ollama uses Chat Completions. Cloud OpenAI uses the Responses API.
    The API is selected by `provider`, not by `reasoning_effort`.

    Args:
        expense_input (Dict[str, str]): The expense input to classify.
        examples (List[Dict[str, Any]], optional): List of example classifications. Defaults to an empty list.
        system_prompt (str, optional): The system prompt to use.
        model_name (str, optional): The name of the model to use.
        temperature (Optional[float], optional): The temperature setting. Not supported by all models.
        response_format (Type[ExpenseOutput], optional): The expected response format.
            Defaults to ExpenseOutput.
        include_categories_in_prompt (bool, optional): If True, appends the category list to the system prompt.
        include_online_search (bool, optional): If True, appends online search results to the user's message.
        reasoning_effort (Optional[str], optional): Ollama thinking control (e.g. "none").
        provider (str, optional): LLM provider. ``ollama`` uses Chat Completions; anything else uses Responses.

    Returns:
        ExpenseOutput: The parsed response containing the classification.
    """
    # If the parameter is True, append the category list to the system prompt
    if include_categories_in_prompt:
        categories_str = create_nested_category_string(response_format)
        # Append the categories to the system prompt
        system_prompt += "\n\n" + categories_str

    if examples is None:
        examples = []

    input_messages: list[dict[str, str]] = []

    for example in examples:
        input_messages.extend(
            [
                {"role": "user", "content": json.dumps(example["input"])},
                {"role": "assistant", "content": example["output"]},
            ]
        )

    user_message_content = json.dumps(expense_input)

    if include_online_search and tavily_client is not None and search_cache is not None:
        expense_name = expense_input.get("Expense_name", "")
        if expense_name:
            search_text = perform_online_search(expense_name, tavily_client, search_cache)
            user_message_content += f"\n\nAdditional Information from Online Search:\n{search_text}"

    input_messages.append({"role": "user", "content": user_message_content})

    parse_kwargs: dict[str, Any] = {"model": model_name}
    if temperature is not None:
        parse_kwargs["temperature"] = temperature

    use_chat_completions = provider == "ollama"
    if use_chat_completions:
        parse_kwargs["messages"] = [{"role": "system", "content": system_prompt}, *input_messages]
        parse_kwargs["response_format"] = response_format
        parse_kwargs["extra_body"] = {"reasoning_effort": reasoning_effort or "none"}
    else:
        parse_kwargs["instructions"] = system_prompt
        parse_kwargs["input"] = input_messages
        parse_kwargs["text_format"] = response_format

    return cast(ExpenseOutput, _parse_structured(llm_client, parse_kwargs, use_chat_completions))


def get_batch_classification(
    expenses: list[dict[str, str]],
    llm_client: OpenAI,
    examples: list[dict[str, Any]] | None = None,
    system_prompt: str = "",
    model_name: str = "gpt-5-mini",
    temperature: float | None = None,
    include_categories_in_prompt: bool = False,
    include_online_search: bool = False,
    tavily_client: TavilyClient | None = None,
    search_cache: SearchCache | None = None,
    reasoning_effort: str | None = None,
    provider: str = "ollama",
) -> ExpenseOutputBatch:
    """Classify a batch of expenses in a single structured LLM call.

    Each item in ``expenses`` must include an ``id`` plus the ExpenseInput fields.
    Local Ollama uses Chat Completions; cloud OpenAI uses the Responses API.
    """
    if include_categories_in_prompt:
        system_prompt += "\n\n" + create_nested_category_string(ExpenseOutput)

    system_prompt += (
        '\n\nYou will receive a JSON object with an "expenses" list. '
        'Each expense has an "id". Classify every expense and return one item per id.'
    )

    if examples is None:
        examples = []

    input_messages: list[dict[str, str]] = []
    for example_index, example in enumerate(examples):
        example_id = f"ex{example_index}"
        input_messages.extend(
            [
                {
                    "role": "user",
                    "content": json.dumps({"expenses": [{"id": example_id, **example["input"]}]}),
                },
                {
                    "role": "assistant",
                    "content": json.dumps({"items": [{"id": example_id, "expense_type": example["output"]}]}),
                },
            ]
        )

    payload_items: list[dict[str, str]] = []
    for expense in expenses:
        item = dict(expense)
        if include_online_search and tavily_client is not None and search_cache is not None:
            expense_name = expense.get("Expense_name", "")
            if expense_name:
                search_text = perform_online_search(expense_name, tavily_client, search_cache)
                item["online_search"] = search_text
        payload_items.append(item)

    input_messages.append({"role": "user", "content": json.dumps({"expenses": payload_items})})

    parse_kwargs: dict[str, Any] = {"model": model_name}
    if temperature is not None:
        parse_kwargs["temperature"] = temperature

    use_chat_completions = provider == "ollama"
    if use_chat_completions:
        parse_kwargs["messages"] = [{"role": "system", "content": system_prompt}, *input_messages]
        parse_kwargs["response_format"] = ExpenseOutputBatch
        parse_kwargs["extra_body"] = {"reasoning_effort": reasoning_effort or "none"}
    else:
        parse_kwargs["instructions"] = system_prompt
        parse_kwargs["input"] = input_messages
        parse_kwargs["text_format"] = ExpenseOutputBatch

    return cast(ExpenseOutputBatch, _parse_structured(llm_client, parse_kwargs, use_chat_completions))


def _classify_single_expense(
    expense_entry: ExpenseEntry,
    llm_client: OpenAI,
    examples: list[dict[str, Any]],
    system_prompt: str,
    model_name: str,
    temperature: float | None,
    response_format: type[ExpenseOutput],
    include_categories_in_prompt: bool,
    include_online_search: bool,
    tavily_client: TavilyClient | None = None,
    search_cache: SearchCache | None = None,
    reasoning_effort: str | None = None,
    provider: str = "ollama",
    classification_cache: ClassificationCache | None = None,
) -> dict[str, Any]:
    """
    Classify a single expense entry.

    Args:
        expense_entry (ExpenseEntry): The expense entry to classify.
        examples (List[Dict[str, Any]]): List of example classifications.
        system_prompt (str): The system prompt to use.
        model_name (str): The name of the model to use.
        temperature (Optional[float]): The temperature setting. Not supported by all models.
        response_format (Type[ExpenseOutput]): The expected response format.
        include_categories_in_prompt (bool): If True, appends the category list to the system prompt.
        include_online_search (bool): If True, appends online search results to the user's message.

    Returns:
        Dict[str, Any]: A dictionary containing the original expense data along with the classification results.
    """
    expense_input = expense_entry.input
    logger.debug(f"Processing expense: {expense_input}")

    amount = _parse_amount(expense_input.Amount, expense_input)
    if amount < 0:
        logger.debug("Skipping classification for negative amount")
        return _expense_result(expense_input, None, None)

    try:
        expense_output = get_classification(
            expense_input=expense_input.model_dump(),
            llm_client=llm_client,
            examples=examples,
            system_prompt=system_prompt,
            model_name=model_name,
            temperature=temperature,
            response_format=response_format,
            include_categories_in_prompt=include_categories_in_prompt,
            include_online_search=include_online_search,
            tavily_client=tavily_client,
            search_cache=search_cache,
            reasoning_effort=reasoning_effort,
            provider=provider,
        )
        if classification_cache is not None:
            classification_cache.put(expense_input.Expense_name, expense_output.category, expense_output.subcategory)
        return _expense_result(expense_input, expense_output.category, expense_output.subcategory)
    except Exception as e:
        logger.error(f"Error processing expense {expense_input}: {e}")
        return _expense_result(expense_input, None, None)


def _classify_expense_batch(
    batch: list[tuple[int, ExpenseEntry]],
    llm_client: OpenAI,
    examples: list[dict[str, Any]],
    system_prompt: str,
    model_name: str,
    temperature: float | None,
    response_format: type[ExpenseOutput],
    include_categories_in_prompt: bool,
    include_online_search: bool,
    tavily_client: TavilyClient | None,
    search_cache: SearchCache | None,
    reasoning_effort: str | None,
    provider: str,
    classification_cache: ClassificationCache,
) -> list[tuple[int, dict[str, Any]]]:
    """Classify a batch of expenses in one LLM call, falling back per row on missing/invalid ids."""
    payload: list[dict[str, str]] = []
    id_to_index_entry: dict[str, tuple[int, ExpenseEntry]] = {}
    for local_id, (orig_idx, entry) in enumerate(batch):
        item_id = str(local_id)
        payload_item = entry.input.model_dump()
        payload_item["id"] = item_id
        payload.append(payload_item)
        id_to_index_entry[item_id] = (orig_idx, entry)

    def fallback_single(orig_idx: int, entry: ExpenseEntry) -> tuple[int, dict[str, Any]]:
        return orig_idx, _classify_single_expense(
            entry,
            llm_client,
            examples,
            system_prompt,
            model_name,
            temperature,
            response_format,
            include_categories_in_prompt,
            include_online_search,
            tavily_client,
            search_cache,
            reasoning_effort,
            provider,
            classification_cache,
        )

    try:
        batch_output = get_batch_classification(
            expenses=payload,
            llm_client=llm_client,
            examples=examples,
            system_prompt=system_prompt,
            model_name=model_name,
            temperature=temperature,
            include_categories_in_prompt=include_categories_in_prompt,
            include_online_search=include_online_search,
            tavily_client=tavily_client,
            search_cache=search_cache,
            reasoning_effort=reasoning_effort,
            provider=provider,
        )
    except Exception as e:
        logger.error(f"Batch classification failed ({len(batch)} expenses): {e}")
        return [fallback_single(orig_idx, entry) for orig_idx, entry in batch]

    returned_by_id: dict[str, Any] = {}
    for item in batch_output.items:
        if item.id in id_to_index_entry and item.id not in returned_by_id:
            returned_by_id[item.id] = item.expense_type

    results: list[tuple[int, dict[str, Any]]] = []
    for item_id, (orig_idx, entry) in id_to_index_entry.items():
        expense_type = returned_by_id.get(item_id)
        if expense_type is None:
            logger.warning(f"Batch response missing or invalid id {item_id}; retrying as a single classification")
            results.append(fallback_single(orig_idx, entry))
            continue
        primary, secondary = _split_expense_type(expense_type)
        classification_cache.put(entry.input.Expense_name, primary, secondary)
        results.append((orig_idx, _expense_result(entry.input, primary, secondary)))
    return results


def classify_expenses(
    df: pd.DataFrame,
    df_examples: pd.DataFrame,
    llm_client: OpenAI,
    system_prompt: str = "",
    model_name: str = "gpt-5-mini",
    temperature: float | None = None,
    response_format: type[ExpenseOutput] = ExpenseOutput,
    include_categories_in_prompt: bool = False,
    include_online_search: bool = False,
    max_workers: int = 2,
    tavily_client: TavilyClient | None = None,
    search_cache: SearchCache | None = None,
    reasoning_effort: str | None = None,
    provider: str = "ollama",
    classification_cache: ClassificationCache | None = None,
    batch_size: int = 10,
) -> pd.DataFrame:
    """
    Classify expenses in the given DataFrame using example data and an LLM.

    Negative amounts are skipped. Cache hits reuse stored Primary/Secondary values.
    Remaining expenses are classified in batches of ``batch_size``; ``max_workers``
    parallelizes those batches.

    Args:
        df (pd.DataFrame): The DataFrame containing expenses to be classified.
        df_examples (pd.DataFrame): The DataFrame containing example expenses for classification.
        system_prompt (str, optional): The system prompt to use for the OpenAI model. Defaults to the value from config_llm.
        model_name (str, optional): The name of the OpenAI model to use. Defaults to the value from config_llm.
        temperature (float, optional): The temperature setting for the OpenAI model. Defaults to the value from config_llm.
        response_format (Type[ExpenseOutput], optional): The expected response format from the OpenAI model.
            Defaults to ExpenseOutput.
        include_categories_in_prompt (bool, optional): If True, appends the category list to the system prompt.
        include_online_search (bool, optional): If True, appends online search results to the user's message.
        max_workers (int, optional): The maximum number of workers for parallel batch processing. Defaults to 2.

    Returns:
        pd.DataFrame: A new DataFrame containing the original expense data along with
        the classification results ('Primary' and 'Secondary' categories).

    Raises:
        Exception: If there's an error during the classification process for an individual expense.
    """
    logger.info("Starting expense classification")

    if classification_cache is None:
        classification_cache = ClassificationCache()
    chunk_size = max(1, batch_size)

    # Get the list of expenses to classify
    expenses = get_list_expenses(df=df, include_output=False)
    logger.debug(f"Got {len(expenses)} expenses to classify")

    # Get the list of example expenses
    examples_list = get_list_expenses(df=df_examples, include_output=True)
    examples = [
        {
            "input": ex.input.model_dump(),
            "output": ex.output.expense_type.value,  # type: ignore[attr-defined]
        }
        for ex in examples_list
        if ex.output is not None
    ]
    logger.debug(f"Got {len(examples)} example expenses")

    results: list[dict[str, Any] | None] = [None] * len(expenses)
    to_classify: list[tuple[int, ExpenseEntry]] = []

    for index, expense_entry in enumerate(expenses):
        expense_input = expense_entry.input
        amount = _parse_amount(expense_input.Amount, expense_input)
        if amount < 0:
            logger.debug("Skipping classification for negative amount")
            results[index] = _expense_result(expense_input, None, None)
            continue

        cached = classification_cache.get(expense_input.Expense_name)
        if cached is not None:
            results[index] = _expense_result(expense_input, cached.get("Primary"), cached.get("Secondary"))
            continue

        to_classify.append((index, expense_entry))

    batches = [to_classify[i : i + chunk_size] for i in range(0, len(to_classify), chunk_size)]  # noqa: E203
    logger.debug(f"Classifying {len(to_classify)} expenses in {len(batches)} batch(es) of up to {chunk_size}")

    if batches:
        with ThreadPoolExecutor(max_workers=max_workers) as executor, tqdm_logging_redirect(desc="Classifying expenses"):
            future_to_batch = {
                executor.submit(
                    _classify_expense_batch,
                    batch,
                    llm_client,
                    examples,
                    system_prompt,
                    model_name,
                    temperature,
                    response_format,
                    include_categories_in_prompt,
                    include_online_search,
                    tavily_client,
                    search_cache,
                    reasoning_effort,
                    provider,
                    classification_cache,
                ): batch
                for batch in batches
            }

            for future in tqdm(as_completed(future_to_batch), total=len(batches), desc="Classifying expenses"):
                batch = future_to_batch[future]
                try:
                    for orig_idx, result in future.result():
                        results[orig_idx] = result
                except Exception as exc:
                    logger.error(f"Expense batch generated an exception: {exc}")
                    for orig_idx, expense_entry in batch:
                        results[orig_idx] = _expense_result(expense_entry.input, None, None)

    classification_results = [
        result if result is not None else _expense_result(expenses[i].input, None, None) for i, result in enumerate(results)
    ]

    # Convert the classification results into a DataFrame
    df_with_output = pd.DataFrame(classification_results)
    logger.debug(f"Created DataFrame with {len(df_with_output)} classified expenses")

    # Ensure the column types match the original DataFrame
    for column in df.columns:
        if column in df_with_output.columns:
            if column == "Day":
                # Convert 'Day' column to datetime with explicit format
                df_with_output[column] = pd.to_datetime(df_with_output[column], format="%d/%m/%Y", errors="coerce")
            else:
                if not df_with_output.empty:
                    df_with_output[column] = df_with_output[column].astype(df[column].dtype)

    logger.debug("Column types adjusted to match original DataFrame")

    # Sort the DataFrame by 'Day', 'Amount', and 'Expense_name'
    if "Day" in df_with_output.columns:
        df_with_output = df_with_output.sort_values(by=["Day", "Amount", "Expense_name"]).reset_index(drop=True)
        logger.debug("DataFrame sorted by 'Day', 'Amount', and 'Expense_name'")

    logger.info("Expense classification completed")
    return df_with_output
