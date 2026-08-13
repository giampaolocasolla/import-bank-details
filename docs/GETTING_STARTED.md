# Getting Started

Practical guidance for working with the codebase.

## Directory Structure

```
import-bank-details/
├── import_bank_details/       # Main package
│   ├── main.py                # Pipeline orchestration, file I/O
│   ├── classification.py      # LLM classification, parallel processing
│   ├── search.py              # Tavily search, caching, rate limiting
│   ├── structured_output.py   # Pydantic models, dynamic ExpenseType enum
│   ├── utils.py               # YAML config loading
│   ├── logger_setup.py        # Logging configuration
│   └── categories.yaml        # Category taxonomy (primary → secondary)
├── tests/                     # Test suite (unit, integration, e2e)
├── data/                      # Input data (one subfolder per bank)
│   └── examples/              # Few-shot CSV + search_cache.json
├── output/                    # Generated Excel files
├── config_bank.yaml           # Bank import settings and column mappings
├── config_llm.yaml            # LLM model selection and system prompt
├── run_tests.sh               # Test runner script
├── pyproject.toml             # Dependencies, tool config
└── .github/workflows/         # CI pipeline (ci.yml, pre-commit.yml)
```

## Key Files

| File | Role |
|---|---|
| `main.py` | Entry point; scans `data/`, imports, processes, classifies, exports |
| `classification.py` | Builds few-shot messages, calls the configured LLM in parallel, returns classified DataFrame |
| `search.py` | `SearchCache` class and `perform_online_search()` for Tavily lookups |
| `structured_output.py` | `ExpenseInput`, `ExpenseOutput`, `ExpenseEntry` models; dynamic `ExpenseType` enum |
| `utils.py` | `load_config()` — generic YAML loader |
| `logger_setup.py` | File + stream logging with timestamped log files in `.log/` |
| `config_bank.yaml` | Per-bank column mappings, date formats, import params, row filters |
| `config_llm.yaml` | Provider, model, Ollama endpoint, reasoning effort, timeout, system prompt |
| `categories.yaml` | Hierarchical expense categories |

## How to Add a New Bank

1. **Create a subfolder** in `data/` with the bank name (e.g., `data/mybank/`).
2. **Place the export file** (CSV or Excel) in that subfolder.
3. **Add a YAML block** to `config_bank.yaml`:
   ```yaml
   mybank:
     import:              # Optional: pandas read_csv kwargs
       sep: ","
     columns_old:         # Column names in the bank's file
       - TransactionDate
       - Description
       - Value
       - Bank             # Use "Bank" as a placeholder — auto-filled with folder name
       - Notes
     columns_new:         # Must always be these five in this order
       - Day
       - Expense_name
       - Amount
       - Bank
       - Comment
     Day: "%Y-%m-%d"      # strftime format for date parsing
     Remove:              # Optional: strings to filter out of Expense_name
       - "Internal Transfer"
   ```
4. **Run the pipeline** — the new bank will be picked up automatically.

If the bank has multiple export formats (like Revolut EN/IT), add a variant config key (e.g., `mybank_variant`) and extend `detect_bank_config()` in `main.py`.

## How to Add a New Category

1. **Edit** `import_bank_details/categories.yaml`:
   ```yaml
   NewPrimary:
     - SubcategoryA
     - SubcategoryB
   ```
   Or add a new subcategory under an existing primary:
   ```yaml
   Housing:
     - Rent
     - NewSubcategory
   ```
2. **No code changes needed** — `ExpenseType` enum is generated dynamically at import time.
3. Optionally add a few-shot example in `data/examples/*.csv` to help the classifier learn the new category.

## How to Modify Classification

- **System prompt** — Edit `config_llm.yaml` → `system_prompt`.
- **Model / provider** — Change `config_llm.yaml` → `llm.provider` (`ollama` or `openai`) and `llm.model_name`.
- **Few-shot examples** — Add/edit rows in `data/examples/*.csv` (columns: Day, Expense_name, Amount, Bank, Comment, Primary, Secondary).
- **Manual classification** — Run with `--skip-classification`, set `llm.provider: openai` without `OPENAI_API_KEY`, or run with local Ollama when the Ollama app is not running (the pipeline exports blank columns instead of hanging).
- **Online search** — Set `TAVILY_API_KEY` to enable search enrichment. Adjust `max_results` in `search.py:perform_online_search()`.
- **Parallel workers** — Change `max_workers` in `config_llm.yaml` → `llm` (default 2 for local Ollama). Workers parallelize classification batches.
- **Batch size** — Change `batch_size` in `config_llm.yaml` → `llm` (default 10 expenses per LLM call).
- **Temperature** — Set `temperature_base` in `config_llm.yaml` → `llm`.

## Testing

Run the test suite:

```sh
./run_tests.sh              # All tests
./run_tests.sh --unit       # Unit tests only
./run_tests.sh --integration # Integration tests only
./run_tests.sh --extra "test_import"  # Pattern match
./run_tests.sh -v           # Verbose mode
```

### Test markers

| Marker | Meaning |
|---|---|
| *(default)* | Unit tests — no markers needed |
| `@pytest.mark.integration` | Integration tests (API calls are mocked) |
| `@pytest.mark.e2e` | End-to-end tests |

### Coverage

Minimum 80% coverage required (`pyproject.toml` → `[tool.coverage.report]` → `fail_under = 80`).

## CI/CD

GitHub Actions runs on every push and PR to `main` (`.github/workflows/ci.yml`):

1. **Ruff** — linting (`ruff check`, including import sorting)
2. **Ruff** — formatting check (`ruff format --check`)
3. **mypy** — static type checking
4. **pytest** — test suite with coverage (`--cov-fail-under=80`)
5. **Codecov** — coverage report upload

A separate `pre-commit.yml` workflow verifies pre-commit hooks.

## Code Quality Settings

| Tool | Setting |
|---|---|
| Line length | 130 characters (Ruff) |
| Ruff lint | `E`, `W`, `F`, `I`, `UP`, `B`, `SIM`, `PT`, `C4` selected in `pyproject.toml` |
| Ruff format | `quote-style = "double"`, `line-length = 130` |
| mypy | Config only in `pyproject.toml`: `python_version = "3.11"`, `disallow_untyped_defs = true`, `warn_return_any = true`, `ignore_missing_imports = true` |
| mypy (tests) | `disallow_untyped_defs = false` |
