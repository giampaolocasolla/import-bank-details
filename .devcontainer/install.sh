#!/bin/sh

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH for the current session
export PATH="$HOME/.local/bin:$PATH"

# Install project dependencies using uv
pip install --upgrade pip
uv sync

# Optional: Check Python and uv versions
echo "Check Python and uv versions"
which python
python --version
which uv
uv --version
uv pip list
