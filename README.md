# Import Bank Details

This project is designed to import and process bank details from various sources. It reads data from CSV or Excel files, processes the data according to specified configurations, and outputs the processed data to an Excel file.

## Features

- Import data from multiple banks (Curve, ING, N26, Revolut)
- Clean and process the data based on configuration
- Output the processed data to an Excel file

## Requirements

- Python 3.11 or higher
- Poetry for dependency management

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

## Usage

Run the main script to process the bank details:

```    sh
poetry run python main.py
```

## Configuration

The configuration file `config.yaml` specifies the import settings and column mappings for different banks. Ensure that this file is correctly set up before running the script.

## License

This project is licensed under the MIT License.