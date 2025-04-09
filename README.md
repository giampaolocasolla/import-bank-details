# Import Bank Details

This project is designed to import, process, and classify bank details from various sources. It reads data from CSV or Excel files, processes the data according to specified configurations, classifies expenses using OpenAI's language model (with optional online search augmentation), and outputs the processed data to an Excel file.

## Features

- **Import Data from Multiple Banks**: Supports Curve, ING, N26, Revolut, and more.
- **Data Cleaning and Processing**: Cleans and processes data based on configurable settings.
- **Expense Classification**:
  - **Primary Classification**: Uses OpenAI's GPT model to categorize expenses.
  - **Enhanced Classification**: Optionally integrates online search results to improve classification accuracy.
- **Online Search Integration**: Fetches additional information from DuckDuckGo to augment expense classification.
- **Output to Excel**: Exports the processed and classified data to an Excel file for easy review and analysis.

## Requirements

- **Python 3.11** or higher
- **Poetry** for dependency management
- **OpenAI API Key**
- **DuckDuckGo Search Integration** (handled via the `duckduckgo_search` Python package)

## Installation

1. **Clone the Repository**:

    ```sh
    git clone https://github.com/yourusername/import-bank-details.git
    cd import-bank-details
    ```

2. **Install Dependencies Using Poetry**:

    ```sh
    poetry install
    ```

3. **Set Up Your API Keys**:
    - **OpenAI API Key**:
        - Create a `.env` file in the project root.
        - Add your OpenAI API key:

          ```
          OPENAI_API_KEY=your_openai_api_key_here
          ```
    - **DuckDuckGo Search**:
        - No additional setup is required as the `duckduckgo_search` package does not require an API key.

## Usage

Run the main script to process and classify the bank details:

```sh
poetry run python main.py
```

### Optional Online Search Enhancement

To enhance expense classification with online search results, enable the `include_online_search` option in your configuration or adjust the function call in your scripts.

Example:

```python
classification = get_classification(
    expense_input=expense_data,
    include_online_search=True
)
```

## Testing

Run the test suite using the provided test script:

```sh
./run_tests.sh
```

The script offers several options:

- `--help`: Show usage information
- `--verbose`, `-v`: Run tests in verbose mode
- `--all`: Run all tests (default)
- `--unit`: Run only unit tests
- `--integration`: Run only integration tests
- `--extra PATTERN`: Run tests matching the given pattern

The test output is color-coded:
- Green: Successful test results
- Red: Failed tests or errors
- Yellow: Informational messages

Example usage:
```sh
# Run all tests
./run_tests.sh

# Run only unit tests in verbose mode
./run_tests.sh --unit -v

# Run tests matching a specific pattern
./run_tests.sh --extra "test_import"
```

## Configuration

- **`config_bank.yaml`**: Specifies the import settings and column mappings for different banks.
- **`config_llm.yaml`**: Contains configuration for the OpenAI language model used for expense classification.

Ensure that these files are correctly set up before running the script.

## Expense Classification

The project uses OpenAI's GPT model to classify expenses into primary and secondary categories. The classification is based on the expense name, amount, and other details. With the optional online search functionality, the classification process can be augmented with additional context fetched from DuckDuckGo, enhancing accuracy for ambiguous or less common expense names.

## License

This project is licensed under the MIT License.
