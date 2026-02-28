# Import Bank Details

[![CI](https://github.com/giampaolocasolla/import-bank-details/actions/workflows/ci.yml/badge.svg)](https://github.com/giampaolocasolla/import-bank-details/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/giampaolocasolla/import-bank-details/branch/main/graph/badge.svg)](https://codecov.io/gh/giampaolocasolla/import-bank-details)

Import, process, and classify bank statements from multiple sources. Reads CSV/Excel exports from various banks, classifies expenses using OpenAI with optional Tavily search enrichment, and outputs a consolidated Excel file.

## Features

- **Multi-bank support** — Curve, ING, N26, Revolut (configuration-driven, easy to extend)
- **Parallel AI classification** — Expenses classified concurrently via OpenAI Responses API with Pydantic structured output
- **Search-augmented classification** — Optional Tavily search enrichment for improved accuracy on ambiguous expenses
- **Few-shot learning** — Example-based prompting from historical classifications
- **Configurable** — Bank mappings, categories, LLM settings all defined in YAML

## Requirements

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) for dependency management
- OpenAI API key
- Tavily API key

## Installation

```sh
git clone https://github.com/giampaolocasolla/import-bank-details.git
cd import-bank-details
uv sync
```

Create a `.env` file in the project root:

```
OPENAI_API_KEY="your_openai_api_key"
TAVILY_API_KEY="your_tavily_api_key"
```

## Usage

```sh
uv run python -m import_bank_details.main
```

Place bank export files in `data/{bank_name}/` and the pipeline will automatically pick up the most recent file from each subfolder.

## Testing

```sh
./run_tests.sh              # All tests
./run_tests.sh --unit       # Unit tests only
./run_tests.sh --integration # Integration tests only
```

## Documentation

See [`docs/`](docs/) for detailed documentation:

- [Architecture](docs/ARCHITECTURE.md) — Tech stack, system overview, data flow, design decisions
- [Abstractions](docs/ABSTRACTIONS.md) — Domain models, configuration formats, classification pipeline
- [Getting Started](docs/GETTING_STARTED.md) — Directory structure, how to extend, testing, CI/CD

## License

This project is licensed under the MIT License.
