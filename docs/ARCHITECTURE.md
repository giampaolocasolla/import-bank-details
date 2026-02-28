# Architecture

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.11+ |
| Package manager | [uv](https://github.com/astral-sh/uv) |
| LLM | OpenAI Responses API (default model: `gpt-5-mini`) |
| Web search | [Tavily](https://tavily.com/) |
| Data processing | Pandas |
| Structured output | Pydantic |
| Excel I/O | openpyxl |
| Configuration | YAML (`PyYAML`) |
| Progress bars | tqdm |

## System Overview

```
 CSV / Excel files (per bank)
        │
        ▼
 ┌──────────────┐
 │   Import      │  Read files, apply bank-specific import params
 └──────┬───────┘
        ▼
 ┌──────────────┐
 │   Process     │  Select/rename columns, parse dates, filter rows
 └──────┬───────┘
        ▼
 ┌──────────────┐
 │   Classify    │  Parallel AI classification (ThreadPoolExecutor)
 │               │  Few-shot examples + optional Tavily search
 │               │  → OpenAI structured output → Pydantic validation
 └──────┬───────┘
        ▼
 ┌──────────────┐
 │   Export      │  Write classified DataFrame to Excel
 └──────────────┘
```

## Data Flow

1. **Import** — `main.py:get_latest_files()` scans `data/` subfolders and picks the most recently modified file per bank. Each file is read with `import_data()` using bank-specific CSV parameters (separator, encoding) or falls back to Excel.

2. **Process** — `process_data()` selects and renames columns per `config_bank.yaml`, removes unwanted rows (e.g., internal transfers), and parses dates. Bank-specific format detection handles variants like Italian vs English Revolut exports (`detect_bank_config()`).

3. **Classify** — `classify_expenses()` sends each expense to OpenAI in parallel via `ThreadPoolExecutor` (default 10 workers). Each request includes few-shot examples from `data/examples/*.csv` and optionally Tavily search results. The LLM returns a structured `ExpenseOutput` (Pydantic model) containing `ExpenseType` — a dynamically generated enum from `categories.yaml`.

4. **Export** — `save_to_excel()` writes the classified DataFrame to `output/` as an Excel file named `{latest_date}_{banks}.xlsx`.

## Key Architectural Decisions

### Configuration-driven bank support
Each bank is defined in `config_bank.yaml` with column mappings (`columns_old` → `columns_new`), date format, optional import parameters (e.g., CSV separator), and optional row-removal filters. Adding a new bank requires only a new YAML block — no code changes.

### Parallel classification with ThreadPoolExecutor
Expenses are classified concurrently (configurable `max_workers`, default 10) to minimize wall-clock time. Progress is tracked via `tqdm`. Each worker is independent — no shared mutable state beyond the thread-safe search cache.

### Structured outputs via Pydantic + OpenAI Responses API
Classification results are parsed directly into `ExpenseOutput` using `openai.responses.parse()`, which enforces the Pydantic schema on the LLM response. This guarantees type-safe, validated output without manual JSON parsing.

### Dynamically generated ExpenseType enum
`structured_output.py:load_expense_type_enum()` reads `categories.yaml` at import time and builds an `Enum` class. This means adding or renaming categories only requires editing the YAML file.

### Thread-safe search result caching with disk persistence
`SearchCache` (in `search.py`) uses a threading lock to protect concurrent reads/writes. Results are persisted to `data/examples/search_cache.json` so repeated runs reuse previous lookups. Rate limiting (min 0.7s between requests) is built in.

### Exponential backoff retry for external API calls
Both OpenAI calls (`get_classification`) and Tavily searches (`perform_online_search`) implement retry loops with exponential backoff (default 3 attempts) to handle transient failures gracefully.

### Modular separation of concerns
| Module | Responsibility |
|---|---|
| `main.py` | Pipeline orchestration, file I/O |
| `classification.py` | LLM interaction, parallel classification |
| `search.py` | Tavily search, caching, rate limiting |
| `structured_output.py` | Pydantic models, dynamic enum generation |
| `utils.py` | YAML config loading |
| `logger_setup.py` | Logging configuration |
