#!/bin/sh

# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Add Poetry to PATH for the current session
export PATH="$HOME/.local/bin:$PATH"

# Install project dependencies using Poetry
pip install --upgrade pip
poetry install --no-interaction --no-ansi

# Optional: Check Python and Poetry versions
echo "Check Python and Poetry versions"
which python
python --version
which poetry
poetry --version
poetry show