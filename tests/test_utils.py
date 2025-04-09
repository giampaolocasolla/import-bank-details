"""Tests for the utils module."""

import os
import tempfile

import pytest
import yaml

from import_bank_details.utils import load_config


def test_load_config(sample_config):
    """Test loading configuration from a YAML file."""
    # Create a temporary file with sample config
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yaml") as f:
        yaml.dump(sample_config, f)
        config_path = f.name

    try:
        # Load the config
        config = load_config(config_path)

        # Check if the config matches the sample config
        assert config == sample_config
        assert "n26" in config
        assert "revolut" in config
        assert "columns_old" in config["n26"]
        assert "columns_new" in config["n26"]
        assert "Day" in config["n26"]
        assert "Remove" in config["revolut"]
    finally:
        # Clean up the temporary file
        if os.path.exists(config_path):
            os.remove(config_path)


def test_load_config_file_not_found():
    """Test load_config function raises FileNotFoundError with invalid path."""
    with pytest.raises(FileNotFoundError):
        load_config("non_existent_file.yaml")


@pytest.mark.skip(reason="utils.load_config error logging needs to be fixed")
def test_load_config_invalid_yaml(tmpdir):
    """Test load_config function raises yaml.YAMLError with invalid YAML."""
    # Create a temporary file with invalid YAML content
    invalid_yaml_file = os.path.join(tmpdir, "invalid.yaml")
    with open(invalid_yaml_file, "w") as f:
        f.write("invalid: yaml: content: :")

    # Check if the function raises yaml.YAMLError
    with pytest.raises(yaml.YAMLError):
        load_config(invalid_yaml_file)
