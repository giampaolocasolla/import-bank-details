# CLAUDE.md

Python CLI that imports multi-bank CSV/Excel statements, classifies expenses via a local Ollama model (or optional OpenAI) with optional Tavily search enrichment, and exports to Excel.

## Commands

```sh
uv sync                                # Install dependencies
uv run python -m import_bank_details.main  # Run pipeline
./run_tests.sh                         # Run all tests
./run_tests.sh --unit                  # Unit tests only
uv run ruff check .                    # Lint (includes import sorting)
uv run ruff format --check .           # Check formatting
uv run mypy .                          # Type check
```

## Key Constraints

- Python 3.11+
- Line length: 130 (Ruff)
- mypy: strict (`disallow_untyped_defs`, `warn_return_any`), tests exempted
- Test coverage: minimum 80%
- Optional API keys: local Ollama is the default classifier (no key); `OPENAI_API_KEY` enables cloud OpenAI when `llm.provider` is `openai`; `TAVILY_API_KEY` enables search enrichment

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — Tech stack, system overview, data flow, design decisions
- [docs/ABSTRACTIONS.md](docs/ABSTRACTIONS.md) — Domain models, configuration formats, classification pipeline
- [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) — Directory structure, how to extend, testing, CI/CD
