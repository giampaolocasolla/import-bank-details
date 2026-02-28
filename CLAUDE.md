# CLAUDE.md

Python CLI that imports multi-bank CSV/Excel statements, classifies expenses via OpenAI (with optional Tavily search enrichment), and exports to Excel.

## Commands

```sh
uv sync                                # Install dependencies
uv run python -m import_bank_details.main  # Run pipeline
./run_tests.sh                         # Run all tests
./run_tests.sh --unit                  # Unit tests only
uv run black --check .                 # Check formatting
uv run flake8 .                        # Lint
uv run isort --check .                 # Check import order
uv run mypy .                          # Type check
```

## Key Constraints

- Python 3.11+
- Line length: 130 (Black + isort + Flake8)
- mypy: strict (`disallow_untyped_defs`, `warn_return_any`), tests exempted
- Test coverage: minimum 80%
- API keys required: `OPENAI_API_KEY`, `TAVILY_API_KEY` (in `.env`)

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — Tech stack, system overview, data flow, design decisions
- [docs/ABSTRACTIONS.md](docs/ABSTRACTIONS.md) — Domain models, configuration formats, classification pipeline
- [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) — Directory structure, how to extend, testing, CI/CD
