# Architecture

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.11+ |
| Package manager | [uv](https://github.com/astral-sh/uv) |
| LLM | OpenAI-compatible API (default: local Ollama `qwen3.5:9b-q8_0`) |
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
 │   Classify    │  Cache by merchant, then batched AI classification
 │               │  Few-shot examples + optional Tavily search
 │               │  → OpenAI-compatible structured output → Pydantic validation
 └──────┬───────┘
        ▼
 ┌──────────────┐
 │   Export      │  Write classified DataFrame to Excel
 └──────────────┘
```

## Data Flow

1. **Import** — `main.py:get_latest_files()` scans `data/` subfolders and picks the most recently modified file per bank. Each file is read with `import_data()` using bank-specific CSV parameters (separator, encoding) or falls back to Excel.

2. **Process** — `process_data()` selects and renames columns per `config_bank.yaml`, removes unwanted rows (e.g., internal transfers), and parses dates. Bank-specific format detection handles variants like Italian vs English Revolut exports (`detect_bank_config()`).

3. **Classify** — `classify_expenses()` looks up each merchant in `ClassificationCache`, then sends remaining expenses to the configured LLM in batches (`batch_size`, default 10). `ThreadPoolExecutor` parallelizes those batches (default 2 workers for local Ollama). Each batch includes few-shot examples from `data/examples/*.csv` and optionally Tavily search results. The LLM returns a structured `ExpenseOutputBatch` (Pydantic) of `ExpenseType` values — a dynamically generated enum from `categories.yaml`. Missing or invalid batch ids fall back to a single `get_classification` call.

4. **Export** — `save_to_excel()` writes the classified DataFrame to `output/` as an Excel file named `{latest_date}_{banks}.xlsx`.

## Key Architectural Decisions

### Configuration-driven bank support
Each bank is defined in `config_bank.yaml` with column mappings (`columns_old` → `columns_new`), date format, optional import parameters (e.g., CSV separator), and optional row-removal filters. Adding a new bank requires only a new YAML block — no code changes.

### Parallel batched classification with ThreadPoolExecutor
Expenses are classified in batches of `batch_size` (default 10). `max_workers` (default 2) parallelizes those batches, not individual rows. Progress is tracked via `tqdm`. Workers share only the thread-safe search and classification caches.

### Structured outputs via Pydantic + OpenAI-compatible APIs
Classification results are parsed into `ExpenseOutput` (single-item fallback) or `ExpenseOutputBatch` (batched requests). The configured `provider` selects the API: local Ollama uses Chat Completions with `reasoning_effort: "none"` so Qwen does not spend tokens on thinking; cloud OpenAI uses `openai.responses.parse()`. Both enforce the Pydantic schema.

### Dynamically generated ExpenseType enum
`structured_output.py:load_expense_type_enum()` reads `categories.yaml` at import time and builds an `Enum` class. This means adding or renaming categories only requires editing the YAML file.

### Thread-safe search result caching with disk persistence
`SearchCache` (in `search.py`) uses a threading lock to protect concurrent reads/writes. Results are persisted to `data/examples/search_cache.json` so repeated runs reuse previous lookups. Rate limiting (min 0.7s between requests) is built in. `ClassificationCache` (in `classification_cache.py`) similarly caches LLM classifications by cleaned merchant name in `data/examples/classification_cache.json`.

### Exponential backoff retry for external API calls
`get_classification` and `get_batch_classification` retry timeouts, 429, and 5xx responses up to 3 times with exponential backoff. Connection failures (Ollama not running) are not retried. Before classifying, `main()` probes Ollama with a short `models.list()` timeout and exports blank columns if the server is down. Tavily searches (`perform_online_search`) still retry transient failures.

### Modular separation of concerns
| Module | Responsibility |
|---|---|
| `main.py` | Pipeline orchestration, file I/O |
| `classification.py` | LLM interaction, cache lookup, batched classification |
| `classification_cache.py` | Merchant-name classification cache |
| `search.py` | Tavily search, caching, rate limiting |
| `structured_output.py` | Pydantic models, dynamic enum generation |
| `utils.py` | YAML config loading |
| `logger_setup.py` | Logging configuration |
