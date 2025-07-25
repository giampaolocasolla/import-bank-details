# %%
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type, cast

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel
from tavily import TavilyClient
from tqdm import tqdm
from tqdm.contrib.logging import tqdm_logging_redirect

from import_bank_details.structured_output import ExpenseEntry, ExpenseInput, ExpenseOutput, ExpenseType
from import_bank_details.utils import load_config

# Get the logger instance
logger = logging.getLogger(__name__)

# Load environment variables from a .env file
load_dotenv()

# client = OpenAI()
# tavily_client = TavilyClient()
client: Optional[OpenAI] = None
tavily_client: Optional[TavilyClient] = None

# Load LLM configuration from YAML file
config_llm = load_config("config_llm.yaml")


def get_openai_client() -> OpenAI:
    """Get the OpenAI client, initializing it if necessary."""
    global client
    if client is None:
        if "OPENAI_API_KEY" not in os.environ:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        client = OpenAI()
    return client


def get_tavily_client() -> TavilyClient:
    """Get the Tavily client, initializing it if necessary."""
    global tavily_client
    if tavily_client is None:
        if "TAVILY_API_KEY" not in os.environ:
            raise ValueError("TAVILY_API_KEY environment variable is not set")
        tavily_client = TavilyClient()
    return tavily_client


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


class SearchCache:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                # Another thread could have created the instance
                # before we acquired the lock, so we check again.
                if not cls._instance:
                    cls._instance = super(SearchCache, cls).__new__(cls)
        return cls._instance

    def __init__(self, max_retries: int = 3, initial_delay: float = 2.0):
        # Ensure __init__ is only run once by checking for an attribute
        if hasattr(self, "_initialized"):
            return
        with self._lock:
            # Check again after acquiring the lock
            if hasattr(self, "_initialized"):
                return
            self.max_retries: int = max_retries
            self.initial_delay: float = initial_delay
            self.last_request_time: float = 0.0
            self.min_request_interval: float = 0.7  # 100 requests/min = ~0.6s/req. Add a small buffer.
            self.cache_lock = threading.Lock()  # Lock specifically for cache read/write
            self._initialized = True

    def get_cache_path(self, custom_path: Optional[Path] = None) -> Path:
        cache_dir = custom_path or Path("data/examples")
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / "search_cache.json"

    def load_cache(self, cache_path: Optional[Path] = None) -> Dict[str, str]:
        path = self.get_cache_path(cache_path)
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning("Cache file corrupted, creating new cache")
        return {}

    def save_cache(self, cache: Dict[str, str], cache_path: Optional[Path] = None):
        path = self.get_cache_path(cache_path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)

    def rate_limit(self) -> None:
        with self._lock:
            current_time = time.time()
            elapsed = current_time - self.last_request_time
            if elapsed < self.min_request_interval:
                time.sleep(self.min_request_interval - elapsed)
            self.last_request_time = time.time()


search_cache = SearchCache()


def perform_online_search(expense_name: str, max_results: int = 2, cache_path: Optional[Path] = None) -> str:
    """
    Search for expense details using Tavily with caching and rate limiting.

    Args:
        expense_name (str): Raw expense name to search for
        max_results (int, optional): Maximum number of search results. Defaults to 2.
        cache_path (Optional[Path], optional): Custom path for cache file. Defaults to None.

    Returns:
        str: JSON-formatted search results string or error message.
            Success: JSON string containing search results
            Failure: Error message as string

    Examples:
        >>> perform_online_search("Coffee Shop Berlin")
        '{"title": "Best Coffee Shops in Berlin", ...}'

        >>> perform_online_search("INVALID", cache_path=Path("/tmp/cache"))
        'Invalid search term'
    """
    logger.info(f"Starting search for expense: '{expense_name}'")

    # Clean input
    texts_to_remove = ["SumUp  *", "PAYPAL *", "LSP*", "CRV*", "PAY.nl*", "UZR*", "luca "]
    cleaned_name = expense_name
    for text in texts_to_remove:
        cleaned_name = cleaned_name.replace(text, "")
    cleaned_name = cleaned_name.strip()

    if not cleaned_name:
        logger.warning(f"Invalid search term after cleaning: {expense_name}")
        return "Invalid search term"

    # Check cache
    cache_key = f"{cleaned_name}:{max_results}"
    with search_cache.cache_lock:
        cache = search_cache.load_cache(cache_path)
        if cache_key in cache:
            logger.info(f"Returning cached results for: {cleaned_name}")
            return cache[cache_key]

    logger.info(f"Performing online search as no cached values for: {cleaned_name}")

    # Implement exponential backoff
    for attempt in range(search_cache.max_retries):
        try:
            search_cache.rate_limit()
            logger.debug(f"Search attempt {attempt + 1} for: {cleaned_name}")

            tavily = get_tavily_client()
            search_results = tavily.search(
                query=cleaned_name,
                search_depth="basic",
                max_results=max_results,
                country="germany",
            )

            search_result_str: str
            if search_results and search_results.get("results"):
                search_result_str = json.dumps(search_results["results"], ensure_ascii=False)
                logger.info(f"Successfully found and cached {len(search_results['results'])} results for: {cleaned_name}")
            else:
                search_result_str = "No results found"
                logger.warning(f"No results found for '{cleaned_name}'")

            # Only cache successful results
            with search_cache.cache_lock:
                cache = search_cache.load_cache(cache_path)
                cache[cache_key] = search_result_str
                search_cache.save_cache(cache, cache_path)
            return search_result_str

        except Exception as e:
            delay = search_cache.initial_delay * (2**attempt)
            logger.warning(f"Search attempt {attempt + 1} failed for '{cleaned_name}': {str(e)}. Retrying in {delay}s")
            time.sleep(delay)

    logger.error(f"Search failed after {search_cache.max_retries} attempts for: {cleaned_name}")
    return "Online search failed after multiple attempts"


def get_classification(
    expense_input: Dict[str, str],
    examples: List[Dict[str, Any]] = [],
    system_prompt: str = config_llm["system_prompt"],
    model_name: str = config_llm["llm"]["model_name"],
    temperature: float = config_llm["llm"]["temperature_base"],
    response_format: Type[ExpenseOutput] = ExpenseOutput,
    include_categories_in_prompt: bool = False,
    include_online_search: bool = False,
) -> ExpenseOutput:
    """
    Get classification for an expense input using OpenAI's chat completion API.

    Args:
        expense_input (Dict[str, str]): The expense input to classify.
        examples (List[Dict[str, Any]], optional): List of example classifications. Defaults to an empty list.
        system_prompt (str, optional): The system prompt to use. Defaults to the value from config_llm.
        model_name (str, optional): The name of the model to use. Defaults to the value from config_llm.
        temperature (float, optional): The temperature setting for the model. Defaults to the value from config_llm.
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

    messages: List[ChatCompletionMessageParam] = [{"role": "system", "content": system_prompt}]

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
        openai_client = get_openai_client()
        response = openai_client.beta.chat.completions.parse(
            model=model_name,
            messages=messages,
            temperature=temperature,
            response_format=response_format,
        )
        return cast(ExpenseOutput, response.choices[0].message.parsed)
    except Exception as e:
        logger.error(f"OpenAI API error: {str(e)}")
        raise


# %%
def _classify_single_expense(
    expense_entry: ExpenseEntry,
    examples: List[Dict[str, Any]],
    system_prompt: str,
    model_name: str,
    temperature: float,
    response_format: Type[ExpenseOutput],
    include_categories_in_prompt: bool,
    include_online_search: bool,
) -> Dict[str, Any]:
    """
    Classify a single expense entry.

    Args:
        expense_entry (ExpenseEntry): The expense entry to classify.
        examples (List[Dict[str, Any]]): List of example classifications.
        system_prompt (str): The system prompt to use.
        model_name (str): The name of the model to use.
        temperature (float): The temperature setting for the model.
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
        logger.info("Skipping classification for negative amount")
        return {
            **expense_input.model_dump(),
            "Primary": None,
            "Secondary": None,
        }

    try:
        expense_output = get_classification(
            expense_input=expense_input.model_dump(),
            examples=examples,
            system_prompt=system_prompt,
            model_name=model_name,
            temperature=temperature,
            response_format=response_format,
            include_categories_in_prompt=include_categories_in_prompt,
            include_online_search=include_online_search,
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
    system_prompt: str = config_llm["system_prompt"],
    model_name: str = config_llm["llm"]["model_name"],
    temperature: float = config_llm["llm"]["temperature_base"],
    response_format: Type[ExpenseOutput] = ExpenseOutput,
    include_categories_in_prompt: bool = False,
    include_online_search: bool = False,
    max_workers: int = 50,
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
    logger.info(f"Got {len(expenses)} expenses to classify")

    # Get the list of example expenses
    examples_list = get_list_expenses(df=df_examples, include_output=True)
    examples = [
        {
            "input": ex.input.model_dump(),
            "output": ex.output.expense_type.value,
        }
        for ex in examples_list
        if ex.output is not None
    ]
    logger.info(f"Got {len(examples)} example expenses")

    # Prepare the list to store classification results
    classification_results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor, tqdm_logging_redirect(desc="Classifying expenses"):
        future_to_expense = {
            executor.submit(
                _classify_single_expense,
                expense_entry,
                examples,
                system_prompt,
                model_name,
                temperature,
                response_format,
                include_categories_in_prompt,
                include_online_search,
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
    logger.info(f"Created DataFrame with {len(df_with_output)} classified expenses")

    # Ensure the column types match the original DataFrame
    for column in df.columns:
        if column in df_with_output.columns:
            if column == "Day":
                # Convert 'Day' column to datetime with explicit format
                df_with_output[column] = pd.to_datetime(df_with_output[column], format="%d/%m/%Y", errors="coerce")
            else:
                if not df_with_output.empty:
                    df_with_output[column] = df_with_output[column].astype(df[column].dtype)

    logger.info("Column types adjusted to match original DataFrame")

    # Sort the DataFrame by 'Day', 'Amount', and 'Expense_name'
    if "Day" in df_with_output.columns:
        df_with_output = df_with_output.sort_values(by=["Day", "Amount", "Expense_name"]).reset_index(drop=True)
        logger.info("DataFrame sorted by 'Day', 'Amount', and 'Expense_name'")

    logger.info("Expense classification completed")
    return df_with_output
