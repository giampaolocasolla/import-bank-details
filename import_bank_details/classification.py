import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Set, Type, cast

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel
from tavily import TavilyClient
from tqdm import tqdm
from tqdm.contrib.logging import tqdm_logging_redirect

from import_bank_details.search import SearchCache, perform_online_search
from import_bank_details.structured_output import ExpenseEntry, ExpenseInput, ExpenseOutput, ExpenseType

logger = logging.getLogger(__name__)


def get_list_expenses(df: pd.DataFrame, include_output: bool = True) -> List[ExpenseEntry]:
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
    expenses: List[ExpenseEntry] = []
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


def create_nested_category_string(response_format: Type[BaseModel]) -> str:
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
    category_dict: Dict[str, Set[str]] = {}
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
    expense_input: Dict[str, str],
    openai_client: OpenAI,
    examples: Optional[List[Dict[str, Any]]] = None,
    system_prompt: str = "",
    model_name: str = "gpt-5-mini",
    temperature: Optional[float] = None,
    response_format: Type[ExpenseOutput] = ExpenseOutput,
    include_categories_in_prompt: bool = False,
    include_online_search: bool = False,
    tavily_client: Optional[TavilyClient] = None,
    search_cache: Optional[SearchCache] = None,
) -> ExpenseOutput:
    """
    Get classification for an expense input using OpenAI's Responses API.

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

    Returns:
        ExpenseOutput: The parsed response from the OpenAI API containing the classification.
    """
    # If the parameter is True, append the category list to the system prompt
    if include_categories_in_prompt:
        categories_str = create_nested_category_string(response_format)
        # Append the categories to the system prompt
        system_prompt += "\n\n" + categories_str

    if examples is None:
        examples = []

    input_messages: List[Dict[str, str]] = []

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

    parse_kwargs: Dict[str, Any] = {
        "model": model_name,
        "instructions": system_prompt,
        "input": input_messages,
        "text_format": response_format,
    }
    if temperature is not None:
        parse_kwargs["temperature"] = temperature

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = openai_client.responses.parse(**parse_kwargs)  # type: ignore[arg-type]
            return cast(ExpenseOutput, response.output_parsed)
        except Exception as e:
            if attempt < max_retries - 1:
                delay = 1.0 * (2**attempt)
                logger.warning(f"OpenAI API attempt {attempt + 1} failed: {str(e)}. Retrying in {delay}s")
                time.sleep(delay)
            else:
                logger.error(f"OpenAI API error after {max_retries} attempts: {str(e)}")
                raise

    # Unreachable: the loop always returns or raises on the last iteration
    raise RuntimeError("Unexpected: retry loop exited without return or raise")


def _classify_single_expense(
    expense_entry: ExpenseEntry,
    openai_client: OpenAI,
    examples: List[Dict[str, Any]],
    system_prompt: str,
    model_name: str,
    temperature: Optional[float],
    response_format: Type[ExpenseOutput],
    include_categories_in_prompt: bool,
    include_online_search: bool,
    tavily_client: Optional[TavilyClient] = None,
    search_cache: Optional[SearchCache] = None,
) -> Dict[str, Any]:
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

    amount_str = expense_input.Amount
    try:
        amount = float(amount_str)
    except ValueError:
        amount = 0
        logger.warning(f"Invalid amount '{amount_str}' for expense: {expense_input}")

    if amount < 0:
        logger.debug("Skipping classification for negative amount")
        return {
            **expense_input.model_dump(),
            "Primary": None,
            "Secondary": None,
        }

    try:
        expense_output = get_classification(
            expense_input=expense_input.model_dump(),
            openai_client=openai_client,
            examples=examples,
            system_prompt=system_prompt,
            model_name=model_name,
            temperature=temperature,
            response_format=response_format,
            include_categories_in_prompt=include_categories_in_prompt,
            include_online_search=include_online_search,
            tavily_client=tavily_client,
            search_cache=search_cache,
        )
        return {
            **expense_input.model_dump(),
            "Primary": expense_output.category,
            "Secondary": expense_output.subcategory,
        }
    except Exception as e:
        logger.error(f"Error processing expense {expense_input}: {e}")
        return {
            **expense_input.model_dump(),
            "Primary": None,
            "Secondary": None,
        }


def classify_expenses(
    df: pd.DataFrame,
    df_examples: pd.DataFrame,
    openai_client: OpenAI,
    system_prompt: str = "",
    model_name: str = "gpt-5-mini",
    temperature: Optional[float] = None,
    response_format: Type[ExpenseOutput] = ExpenseOutput,
    include_categories_in_prompt: bool = False,
    include_online_search: bool = False,
    max_workers: int = 10,
    tavily_client: Optional[TavilyClient] = None,
    search_cache: Optional[SearchCache] = None,
) -> pd.DataFrame:
    """
    Classify expenses in the given DataFrame using example data and OpenAI's language model.

    This function processes each expense in the input DataFrame, classifies it using the
    provided example data and the OpenAI model, and returns a new DataFrame with the
    classification results. This version uses parallel processing to speed up the classification.

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
        max_workers (int, optional): The maximum number of workers for parallel processing. Defaults to 50.

    Returns:
        pd.DataFrame: A new DataFrame containing the original expense data along with
        the classification results ('Primary' and 'Secondary' categories).

    Raises:
        Exception: If there's an error during the classification process for an individual expense.
    """
    logger.info("Starting expense classification")

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

    # Prepare the list to store classification results
    classification_results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor, tqdm_logging_redirect(desc="Classifying expenses"):
        future_to_expense = {
            executor.submit(
                _classify_single_expense,
                expense_entry,
                openai_client,
                examples,
                system_prompt,
                model_name,
                temperature,
                response_format,
                include_categories_in_prompt,
                include_online_search,
                tavily_client,
                search_cache,
            ): expense_entry
            for expense_entry in expenses
        }

        for future in tqdm(as_completed(future_to_expense), total=len(expenses), desc="Classifying expenses"):
            try:
                result = future.result()
                classification_results.append(result)
            except Exception as exc:
                expense_input = future_to_expense[future].input
                logger.error(f"Expense {expense_input} generated an exception: {exc}")
                classification_results.append(
                    {
                        **expense_input.model_dump(),
                        "Primary": None,
                        "Secondary": None,
                    }
                )

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
