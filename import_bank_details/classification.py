# %%
import json
import logging
import os
from typing import Any, Dict, List, Set

import pandas as pd
from dotenv import load_dotenv
from duckduckgo_search import DDGS
from openai import OpenAI
from pydantic import BaseModel

from import_bank_details.structured_output import ExpenseEntry, ExpenseInput, ExpenseOutput, ExpenseType
from import_bank_details.utils import load_config

# Get the logger instance
logger = logging.getLogger(__name__)

# Load environment variables from a .env file
load_dotenv()

# Verify that the OpenAI API key environment variable is set
if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

client = OpenAI()

# Load LLM configuration from YAML file
config_llm = load_config("config_llm.yaml")


# %%
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


def create_nested_category_string(response_format: BaseModel) -> str:
    """
    Generates a nested list of primary and secondary categories from the BaseModel's schema.

    Args:
        response_format (BaseModel): The Pydantic BaseModel containing the schema.

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


def perform_online_search(expense_name: str, region: str = "de-de", max_results: int = 2) -> str:
    """
    Perform an online search using DuckDuckGo and return the search results as a string.

    Args:
        expense_name (str): The name of the expense to search for.
        region (str, optional): The region to search in. Defaults to "de-de".
        max_results (int, optional): The maximum number of search results to return. Defaults to 2.

    Returns:
        str: Concatenated search results or an error message if the search fails.
    """
    # List of texts to remove from the expense_name
    texts_to_remove = ["SumUp  *", "PAYPAL *", "LSP*", "CRV*", "PAY.nl*", "UZR*", "luca "]

    # Remove unwanted texts from the expense_name
    for text in texts_to_remove:
        expense_name = expense_name.replace(text, "")

    # Strip extra whitespace after removal
    expense_name = expense_name.strip()

    try:
        with DDGS() as ddgs:
            search_results = list(ddgs.text(expense_name, region=region, max_results=max_results))
            # Convert each search result dict to a JSON string
            search_results_str = "\n".join([json.dumps(result, ensure_ascii=False) for result in search_results])
            return search_results_str
    except Exception as e:
        logger.error(f"Online search failed: {str(e)}")
        # Return a fallback message if the search fails
        return "[Online search failed to retrieve additional information.]"


def get_classification(
    expense_input: Dict[str, str],
    examples: List[Dict[str, Any]] = [],
    system_prompt: str = config_llm["system_prompt"],
    model_name: str = config_llm["llm"]["model_name"],
    temperature: float = config_llm["llm"]["temperature_base"],
    response_format: BaseModel = ExpenseOutput,
    include_categories_in_prompt: bool = False,
    include_online_search: bool = False,
) -> BaseModel:
    """
    Get classification for an expense input using OpenAI's chat completion API.

    Args:
        expense_input (Dict[str, str]): The expense input to classify.
        examples (List[Dict[str, Any]], optional): List of example classifications. Defaults to an empty list.
        system_prompt (str, optional): The system prompt to use. Defaults to the value from config_llm.
        model_name (str, optional): The name of the model to use. Defaults to the value from config_llm.
        temperature (float, optional): The temperature setting for the model. Defaults to the value from config_llm.
        response_format (BaseModel, optional): The expected response format. Defaults to ExpenseOutput.
        include_categories_in_prompt (bool, optional): If True, appends the category list to the system prompt.
        include_online_search (bool, optional): If True, appends online search results to the user's message.

    Returns:
        BaseModel: The parsed response from the OpenAI API containing the classification.
    """
    # If the parameter is True, append the category list to the system prompt
    if include_categories_in_prompt:
        categories_str = create_nested_category_string(response_format)
        # Append the categories to the system prompt
        system_prompt += "\n\n" + categories_str

    messages = [{"role": "system", "content": system_prompt}]

    for example in examples:
        messages.extend(
            [
                {"role": "user", "content": json.dumps(example["input"])},
                {"role": "assistant", "content": example["output"]},
            ]
        )

    user_message_content = json.dumps(expense_input)

    if include_online_search:
        expense_name = expense_input.get("Expense_name", "")
        if expense_name:
            search_text = perform_online_search(expense_name)
            user_message_content += f"\n\nAdditional Information from Online Search:\n{search_text}"

    messages.append({"role": "user", "content": user_message_content})

    try:
        response = client.beta.chat.completions.parse(
            model=model_name,
            messages=messages,
            temperature=temperature,
            response_format=response_format,
        )
        return response.choices[0].message.parsed
    except Exception as e:
        logger.error(f"OpenAI API error: {str(e)}")
        raise


# %%
def classify_expenses(
    df: pd.DataFrame,
    df_examples: pd.DataFrame,
    system_prompt: str = config_llm["system_prompt"],
    model_name: str = config_llm["llm"]["model_name"],
    temperature: float = config_llm["llm"]["temperature_base"],
    response_format: BaseModel = ExpenseOutput,
    include_categories_in_prompt: bool = False,
    include_online_search: bool = False,
) -> pd.DataFrame:
    """
    Classify expenses in the given DataFrame using example data and OpenAI's language model.

    This function processes each expense in the input DataFrame, classifies it using the
    provided example data and the OpenAI model, and returns a new DataFrame with the
    classification results.

    Args:
        df (pd.DataFrame): The DataFrame containing expenses to be classified.
        df_examples (pd.DataFrame): The DataFrame containing example expenses for classification.
        system_prompt (str, optional): The system prompt to use for the OpenAI model. Defaults to the value from config_llm.
        model_name (str, optional): The name of the OpenAI model to use. Defaults to the value from config_llm.
        temperature (float, optional): The temperature setting for the OpenAI model. Defaults to the value from config_llm.
        response_format (BaseModel, optional): The expected response format from the OpenAI model. Defaults to ExpenseOutput.
        include_categories_in_prompt (bool, optional): If True, appends the category list to the system prompt.
        include_online_search (bool, optional): If True, appends online search results to the user's message.

    Returns:
        pd.DataFrame: A new DataFrame containing the original expense data along with
        the classification results ('Primary' and 'Secondary' categories).

    Raises:
        Exception: If there's an error during the classification process for an individual expense.
    """
    logger.info("Starting expense classification")

    # Get the list of expenses to classify
    expenses = get_list_expenses(df=df, include_output=False)
    logger.info(f"Got {len(expenses)} expenses to classify")

    # Get the list of example expenses
    examples = get_list_expenses(df=df_examples, include_output=True)
    logger.info(f"Got {len(examples)} example expenses")

    # Prepare the list to store classification results
    classification_results = []

    for expense_entry in expenses:
        expense_input = expense_entry.input
        logger.debug(f"Processing expense: {expense_input}")

        # Check if the 'Amount' is negative
        amount_str = expense_input.Amount
        try:
            amount = float(amount_str)
        except ValueError:
            amount = 0  # Default to 0 if amount is not a valid number
            logger.warning(f"Invalid amount '{amount_str}' for expense: {expense_input}")

        if amount < 0:
            logger.info("Skipping classification for negative amount")
            # Append the expense with 'Primary' and 'Secondary' set to None
            classification_results.append(
                {
                    **expense_input.model_dump(),
                    "Primary": None,
                    "Secondary": None,
                }
            )
            continue  # Skip to the next expense

        try:
            expense_output = get_classification(
                expense_input=expense_input.model_dump(),
                examples=[
                    {
                        "input": ex.input.model_dump(),
                        "output": ex.output.expense_type.value,
                    }
                    for ex in examples
                    if ex.output is not None
                ],
                system_prompt=system_prompt,
                model_name=model_name,
                temperature=temperature,
                response_format=response_format,
                include_categories_in_prompt=include_categories_in_prompt,
                include_online_search=include_online_search,
            )
            classification_results.append(
                {
                    **expense_input.model_dump(),
                    "Primary": expense_output.category,
                    "Secondary": expense_output.subcategory,
                }
            )
            logger.debug(f"Classified expense as {expense_output.category}, {expense_output.subcategory}")
        except Exception as e:
            logger.error(f"Error processing expense {expense_input}: {e}")
            # Handle parsing error by appending None or default values
            classification_results.append(
                {
                    **expense_input.model_dump(),
                    "Primary": None,
                    "Secondary": None,
                }
            )

    # Convert the classification results into a DataFrame
    df_with_output = pd.DataFrame(classification_results)
    logger.info(f"Created DataFrame with {len(df_with_output)} classified expenses")

    # Ensure the column types match the original DataFrame
    for column in df.columns:
        if column in df_with_output.columns:
            if column == "Day":
                # Convert 'Day' column to datetime with explicit format
                df_with_output[column] = pd.to_datetime(df_with_output[column], format="%d/%m/%Y", errors="coerce")
            else:
                df_with_output[column] = df_with_output[column].astype(df[column].dtype)

    logger.info("Column types adjusted to match original DataFrame")

    logger.info("Expense classification completed")
    return df_with_output
