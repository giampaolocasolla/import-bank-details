# Import Bank Details

[![CI](https://github.com/giampaolocasolla/import-bank-details/actions/workflows/ci.yml/badge.svg)](https://github.com/giampaolocasolla/import-bank-details/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/giampaolocasolla/import-bank-details/branch/main/graph/badge.svg)](https://codecov.io/gh/giampaolocasolla/import-bank-details)

This project is designed to import, process, and classify bank details from various sources. It reads data from CSV or Excel files, processes the data according to specified configurations, classifies expenses using OpenAI's language model (with optional online search augmentation), and outputs the processed data to an Excel file.

## Features

- **Import Data from Multiple Banks**: Supports Curve, ING, N26, Revolut, and more.
- **Data Cleaning and Processing**: Cleans and processes data based on configurable settings.
- **Parallel Expense Classification**:
  - Classifies expenses in parallel to significantly speed up processing time.
  - The number of parallel workers can be configured in `import_bank_details/main.py`.
- **Primary Classification**: Uses OpenAI's GPT model to categorize expenses.
- **Enhanced Classification**: Optionally integrates online search results to improve classification accuracy.
- **Online Search Integration**: Fetches additional information from Tavily to augment expense classification.
- **Output to Excel**: Exports the processed and classified data to an Excel file for easy review and analysis.

## Requirements

- **Python 3.11** or higher
- **uv** for dependency management
- **OpenAI API Key**
- **Tavily API Key**

## Installation

1. **Clone the Repository**:

    ```sh
    git clone https://github.com/yourusername/import-bank-details.git
    cd import-bank-details
    ```

2. **Install uv (if not already installed)**:

    ```sh
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

3. **Install Dependencies Using uv**:

    ```sh
    uv sync
    ```

4. **Set Up Your API Keys**:
    - Create a `.env` file in the project root.
    - Add your OpenAI and Tavily API keys:

      ```
      OPENAI_API_KEY="your_openai_api_key"
      TAVILY_API_KEY="your_tavily_api_key"
      ```

## Usage

Run the main script to process and classify the bank details:

```sh
uv run python -m import_bank_details.main
```

### Classification with Online Search

The application can perform online searches to gather additional context for classifying expenses. This feature is powered by Tavily and can be enabled by setting `include_online_search=True` when calling `classify_expenses`.
The search results are cached to improve performance and reduce redundant searches.
The cache is stored in `data/examples/search_cache.json`.

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

## CI/CD Pipeline

This project uses GitHub Actions for Continuous Integration and Continuous Deployment:

- **Automated Tests**: All tests are automatically run on each push and pull request.
- **Code Quality Checks**:
  - Code formatting with Black
  - Linting with Flake8
  - Import sorting with isort
  - Type checking with mypy
  - Pre-commit hook verification
- **Test Coverage**: Coverage reports are generated and uploaded to Codecov.

You can see the status of these checks on any pull request or in the Actions tab of the GitHub repository.

## Configuration

- **`config_bank.yaml`**: Specifies the import settings and column mappings for different banks.
- **`config_llm.yaml`**: Contains configuration for the OpenAI language model used for expense classification.

Ensure that these files are correctly set up before running the script.

## Expense Classification

The project uses OpenAI's GPT model to classify expenses into primary and secondary categories. The classification is based on the expense name, amount, and other details. With the optional online search functionality, the classification process can be augmented with additional context fetched from Tavily, enhancing accuracy for ambiguous or less common expense names.

### Classification Examples

The `data/examples` directory contains a CSV file with example classifications. This file is used to provide few-shot examples to the language model, which improves the accuracy of the classification.

## License

This project is licensed under the MIT License.
