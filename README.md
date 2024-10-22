# Import Bank Details

This project is designed to import, process, and classify bank details from various sources. It reads data from CSV or Excel files, processes the data according to specified configurations, classifies expenses using OpenAI's language model, and outputs the processed data to an Excel file.

## Features

- Import data from multiple banks (Curve, ING, N26, Revolut)
- Clean and process the data based on configuration
- Classify expenses using OpenAI's GPT model
- Output the processed and classified data to an Excel file

## Requirements

- Python 3.11 or higher
- Poetry for dependency management
- OpenAI API key

## Installation

1. Clone the repository:

```sh
git clone https://github.com/yourusername/import-bank-details.git
cd import-bank-details
```

2. Install dependencies using Poetry:

```sh
poetry install
```

3. Set up your OpenAI API key:
   - Create a `.env` file in the project root
   - Add your OpenAI API key: `OPENAI_API_KEY=your_api_key_here`

## Usage

Run the main script to process and classify the bank details:

```sh
poetry run python main.py
```

## Configuration

- `config_bank.yaml`: Specifies the import settings and column mappings for different banks.
- `config_llm.yaml`: Contains configuration for the OpenAI language model used for expense classification.

Ensure that these files are correctly set up before running the script.

## Expense Classification

The project uses OpenAI's GPT model to classify expenses into primary and secondary categories. The classification is based on the expense name, amount, and other details. You can customize the classification categories in the `structured_output.py` file.

## License

This project is licensed under the MIT License.
