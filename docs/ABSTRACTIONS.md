# Abstractions

Key domain concepts and data models the project is invested in.

## Bank Configuration Model

Defined in `config_bank.yaml`. Each bank is a top-level YAML key with:

| Field | Purpose | Example |
|---|---|---|
| `import` | Pandas `read_csv` kwargs (separator, encoding, etc.) | `sep: ";"` |
| `columns_old` | Original column names in the bank's export | `["Date", "Bénéficiaire", "Montant", "Bank", "Communication"]` |
| `columns_new` | Standardized column names | `["Day", "Expense_name", "Amount", "Bank", "Comment"]` |
| `Day` | `strftime` format string for date parsing | `"%d/%m/%Y"` |
| `Remove` | (optional) Strings to filter out of `Expense_name` | `["Giampaolo Casolla"]` (internal transfers) |

The `Bank` column is auto-filled with the subfolder name. If `Bank` appears in `columns_old`, the folder name is assigned before renaming.

**Supported banks:** Curve, ING, N26, Revolut (English and Italian variants).

Revolut has automatic format detection — `detect_bank_config()` in `main.py` reads the CSV header and selects `revolut` or `revolut_it` configuration.

## Category Taxonomy

Defined in `import_bank_details/categories.yaml`. A hierarchical YAML structure:

```yaml
Housing:
  - Rent
  - Internet
  - Furniture
  # ...

Transport:
  - Fuel
  - PublicTransport
  # ...
```

**Primary categories:** Housing, Transport, Groceries, Travel, Out, Leisure, Health, Gifts, Clothing, Fees, OtherExpenses.

Each primary category has a list of secondary categories (subcategories).

### ExpenseType Enum

`structured_output.py:load_expense_type_enum()` reads the YAML at import time and dynamically generates a Python `Enum`:

- Each enum member name is `SUBCATEGORY_UPPER` (e.g., `RENT`, `FUEL`)
- Each enum value is `"Primary, Secondary"` (e.g., `"Housing, Rent"`, `"Transport, Fuel"`)
- Collisions are resolved by prefixing with the primary category (e.g., `LEISURE_LEISURE` for `Leisure > Leisure` vs `OUT_LEISURE` for `Out > Leisure`)

## Expense Data Models

Defined in `structured_output.py` using Pydantic:

### ExpenseInput
```
Day: str           # "dd/mm/yyyy"
Expense_name: str  # Raw description from bank
Amount: str        # Numeric string (e.g., "42.50")
Bank: str          # Bank/folder name
Comment: str       # Additional info (payment type, reference, etc.)
```

### ExpenseOutput
```
expense_type: ExpenseType   # The dynamically generated enum
├── .category    → str      # Primary category (property)
└── .subcategory → str      # Secondary category (property)
```

Validation via `@field_validator` ensures only valid `ExpenseType` members are accepted.

### ExpenseEntry
```
input: ExpenseInput
output: Optional[ExpenseOutput]   # None before classification
```

### Pipeline flow
`ExpenseInput` → classification → `ExpenseOutput` → combined into `ExpenseEntry`

## Classification Pipeline

1. **Build few-shot examples** — Convert `data/examples/*.csv` rows into `ExpenseEntry` objects with known outputs, then serialize as `{"input": {...}, "output": "Primary, Secondary"}` pairs.

2. **Optional search enrichment** — If `include_online_search=True`, call `perform_online_search()` which queries Tavily for the expense name (with a Germany country filter, falling back without), caches results, and appends them to the user message.

3. **LLM call** — Send system prompt (from `config_llm.yaml`) + optional category list + few-shot messages + user message to `openai.responses.parse()`. The response is constrained to the `ExpenseOutput` Pydantic schema.

4. **Validation** — Pydantic validates that the returned `expense_type` is a valid `ExpenseType` enum member.

Negative amounts (income/refunds) skip classification entirely.

## LLM Configuration

Defined in `config_llm.yaml`:

```yaml
llm:
  model_name: "gpt-5-mini"
  timeout: 180

system_prompt: "You are an helpful assistant that classifies expenses into categories and subcategories."
```

The system prompt is augmented at runtime with the full nested category list when `include_categories_in_prompt=True`.

## Search and Caching

### SearchCache class (`search.py`)

- **Thread-safe**: All access goes through a `threading.Lock`
- **Lazy-loaded**: Disk cache is read on first access, not at init
- **Disk-persistent**: Written to `data/examples/search_cache.json` after every new entry
- **Rate-limited**: Minimum 0.7s between Tavily API requests
- **Retry-ready**: Configurable `max_retries` (default 3) and `initial_delay` (default 2.0s) for exponential backoff

### Input cleaning

Before searching, common prefixes are stripped: `SumUp  *`, `PAYPAL *`, `LSP*`, `CRV*`, `PAY.nl*`, `UZR*`, `luca `.

### Cache key format

`"{cleaned_name}:{max_results}"` — so different `max_results` values produce separate cache entries.
