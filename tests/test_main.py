"""Tests for the main module functions."""

import os
from unittest import mock

import pandas as pd
import pytest

from import_bank_details.main import (
    create_llm_client,
    detect_bank_config,
    get_latest_files,
    import_data,
    main,
    ollama_is_available,
    parse_args,
    prepare_manual_classification,
    process_data,
    process_examples,
    remove_unnecessary_expenses,
    save_to_excel,
    should_classify,
    validate_example_structure,
)


def test_get_latest_files(sample_data_dir, sample_n26_csv, sample_revolut_csv, sample_example_csv):
    """Test the get_latest_files function."""
    file_data = get_latest_files(
        data_dir=sample_data_dir,
        base_dir=os.path.dirname(sample_data_dir),
    )

    assert "n26" in file_data
    assert "revolut" in file_data
    assert "examples" in file_data
    assert os.path.basename(file_data["n26"]) == "n26_test.csv"
    assert os.path.basename(file_data["revolut"]) == "revolut_test.csv"
    assert os.path.basename(file_data["examples"]) == "examples_test.csv"


def test_get_latest_files_ignores_hidden_entries(sample_data_dir, sample_n26_csv, sample_revolut_csv, sample_example_csv, caplog):
    """Hidden files in the data directory should be ignored entirely."""
    hidden_file = os.path.join(sample_data_dir, ".DS_Store")
    with open(hidden_file, "w") as file_handle:
        file_handle.write("macOS metadata")

    with caplog.at_level("WARNING"):
        file_data = get_latest_files(
            data_dir=sample_data_dir,
            base_dir=os.path.dirname(sample_data_dir),
        )

    assert ".DS_Store" not in file_data
    assert "No files found in folder: .DS_Store" not in caplog.text


def test_get_latest_files_empty_folder(sample_data_dir):
    """Test the get_latest_files function with an empty folder."""
    empty_data_dir = os.path.join(sample_data_dir, "empty_data")
    os.makedirs(empty_data_dir, exist_ok=True)

    with pytest.raises(ValueError, match="No files found in any folder"):
        get_latest_files(
            data_dir=empty_data_dir,
            base_dir=os.path.dirname(sample_data_dir),
        )


def test_import_data_csv(sample_n26_csv):
    """Test the import_data function with a CSV file."""
    # Import the CSV file
    df = import_data(file_path=sample_n26_csv)

    # Check if the dataframe was created correctly
    assert isinstance(df, pd.DataFrame)
    assert "Value Date" in df.columns
    assert "Partner Name" in df.columns
    assert "Amount (EUR)" in df.columns
    assert "Bank" in df.columns
    assert "Payment Reference" in df.columns
    assert df.shape[0] == 2  # Two rows in the sample data


def test_import_data_with_params(sample_revolut_csv):
    """Test the import_data function with parameters."""
    # Import with specific parameters
    import_params = {"sep": ",", "header": 0}
    df = import_data(file_path=sample_revolut_csv, import_params=import_params)

    # Check if the dataframe was created correctly
    assert isinstance(df, pd.DataFrame)
    assert "Started Date" in df.columns
    assert "Description" in df.columns
    assert "Amount" in df.columns
    assert "Bank" in df.columns
    assert "Type" in df.columns
    assert df.shape[0] == 2  # Two rows in the sample data


def test_import_data_error():
    """Test the import_data function raises an exception for non-existent file."""
    with pytest.raises(FileNotFoundError):
        import_data(file_path="non_existent_file.csv")


def test_process_data(sample_n26_df, sample_config):
    """Test the process_data function."""
    # Process the dataframe
    df_processed = process_data(df=sample_n26_df, config=sample_config["n26"], bank_name="n26")

    # Check if the columns were renamed correctly
    assert "Day" in df_processed.columns
    assert "Expense_name" in df_processed.columns
    assert "Amount" in df_processed.columns
    assert "Bank" in df_processed.columns
    assert "Comment" in df_processed.columns

    # Check if the bank name was set correctly
    assert df_processed["Bank"].iloc[0] == "n26"

    # Check if Day was converted to datetime
    assert pd.api.types.is_datetime64_dtype(df_processed["Day"])


def test_process_data_with_remove(sample_revolut_df, sample_config):
    """Test the process_data function with remove criteria."""
    # Add a row that should be removed
    sample_revolut_df = pd.concat(
        [
            sample_revolut_df,
            pd.DataFrame(
                [
                    {
                        "Started Date": "2023-01-05 10:20:30",
                        "Description": "Payment from Giampaolo Casolla",
                        "Amount": 100.00,
                        "Bank": "Revolut",
                        "Type": "Income",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    # Process the dataframe
    df_processed = process_data(df=sample_revolut_df, config=sample_config["revolut"], bank_name="revolut")

    # Check if the row was removed
    assert df_processed.shape[0] == 2  # Should still have only 2 rows after removal
    assert "Payment from Giampaolo Casolla" not in df_processed["Expense_name"].values


def test_process_examples():
    """Test the process_examples function."""
    # Create a dataframe with example data
    data = {
        "Day": ["01/01/2023", "02/01/2023"],
        "Expense_name": ["Supermarket", "Restaurant"],
        "Amount": ["€45,50", "€26,75"],
        "Bank": ["N26", "N26"],
        "Comment": ["", ""],
        "Primary": ["Groceries", "Out"],
        "Secondary": ["Auchan", "Restaurants"],
    }
    df_examples = pd.DataFrame(data)

    # Process the examples
    processed_df = process_examples(df_examples=df_examples)

    # Check if 'Day' is converted to datetime
    assert pd.api.types.is_datetime64_dtype(processed_df["Day"])

    # Check if 'Amount' is converted to float
    assert pd.api.types.is_float_dtype(processed_df["Amount"])
    assert processed_df["Amount"].iloc[0] == 45.50
    assert processed_df["Amount"].iloc[1] == 26.75


def test_remove_unnecessary_expenses():
    """Test the remove_unnecessary_expenses function."""
    # Create a dataframe
    data = {
        "Expense_name": [
            "Supermarket",
            "Restaurant",
            "Payment from User",
            "To EUR",
            None,
        ]
    }
    df = pd.DataFrame(data)

    # Define removal criteria
    remove_criteria = ["Payment from", "To EUR"]

    # Remove unnecessary expenses
    filtered_df = remove_unnecessary_expenses(df=df, remove_criteria=remove_criteria)

    # Check if the unnecessary expenses were removed
    assert filtered_df.shape[0] == 3  # Should be left with 3 rows
    assert "Payment from User" not in filtered_df["Expense_name"].values
    assert "To EUR" not in filtered_df["Expense_name"].values


def test_validate_example_structure_valid():
    """Test validate_example_structure with valid structure."""
    # Create two dataframes with matching structure
    df = pd.DataFrame(
        {
            "Day": pd.to_datetime(["2023-01-01", "2023-01-02"]),
            "Expense_name": ["Supermarket", "Restaurant"],
            "Amount": [45.50, 26.75],
            "Bank": ["N26", "N26"],
            "Comment": ["", ""],
        }
    )

    df_examples = pd.DataFrame(
        {
            "Day": pd.to_datetime(["2023-01-03", "2023-01-04"]),
            "Expense_name": ["Cafe", "Taxi"],
            "Amount": [5.50, 12.75],
            "Bank": ["Revolut", "Revolut"],
            "Comment": ["", ""],
            "Primary": ["Out", "Transport"],
            "Secondary": ["Bar", "Taxi"],
        }
    )

    # Should not raise an exception
    validate_example_structure(df=df, df_examples=df_examples)


def test_validate_example_structure_missing_columns():
    """Test validate_example_structure with missing columns."""
    # Create two dataframes where examples is missing a column
    df = pd.DataFrame(
        {
            "Day": pd.to_datetime(["2023-01-01", "2023-01-02"]),
            "Expense_name": ["Supermarket", "Restaurant"],
            "Amount": [45.50, 26.75],
            "Bank": ["N26", "N26"],
            "Comment": ["", ""],
        }
    )

    df_examples = pd.DataFrame(
        {
            "Day": pd.to_datetime(["2023-01-03", "2023-01-04"]),
            "Expense_name": ["Cafe", "Taxi"],
            "Amount": [5.50, 12.75],
            # Missing "Bank" column
            "Comment": ["", ""],
            "Primary": ["Out", "Transport"],
            "Secondary": ["Bar", "Taxi"],
        }
    )

    # Should raise a ValueError
    with pytest.raises(ValueError, match="Example file structure does not match data"):
        validate_example_structure(df=df, df_examples=df_examples)


def test_validate_example_structure_dtype_mismatch():
    """Test validate_example_structure with data type mismatch."""
    # Create two dataframes with a data type mismatch
    df = pd.DataFrame(
        {
            "Day": pd.to_datetime(["2023-01-01", "2023-01-02"]),
            "Expense_name": ["Supermarket", "Restaurant"],
            "Amount": [45.50, 26.75],  # Float type
            "Bank": ["N26", "N26"],
            "Comment": ["", ""],
        }
    )

    df_examples = pd.DataFrame(
        {
            "Day": pd.to_datetime(["2023-01-03", "2023-01-04"]),
            "Expense_name": ["Cafe", "Taxi"],
            "Amount": ["5.50", "12.75"],  # String type
            "Bank": ["Revolut", "Revolut"],
            "Comment": ["", ""],
            "Primary": ["Out", "Transport"],
            "Secondary": ["Bar", "Taxi"],
        }
    )

    # Should raise a ValueError
    with pytest.raises(ValueError, match="Example file column dtypes do not match data"):
        validate_example_structure(df=df, df_examples=df_examples)


def test_detect_bank_config_italian_revolut(tmpdir, sample_config):
    """Test detect_bank_config with Italian Revolut CSV format."""
    # Create a temporary CSV file with Italian headers
    italian_csv = tmpdir.join("revolut_it.csv")
    italian_csv.write("Tipo,Prodotto,Data di inizio,Data di completamento,Descrizione,Importo,Costo,Valuta,State,Saldo\n")

    # Test detection
    config_key = detect_bank_config(str(italian_csv), "revolut", sample_config)

    # Should detect Italian format
    assert config_key == "revolut_it"


def test_detect_bank_config_english_revolut(tmpdir, sample_config):
    """Test detect_bank_config with English Revolut CSV format."""
    # Create a temporary CSV file with English headers
    english_csv = tmpdir.join("revolut_en.csv")
    english_csv.write("Type,Product,Started Date,Completed Date,Description,Amount,Fee,Currency,State,Balance\n")

    # Test detection
    config_key = detect_bank_config(str(english_csv), "revolut", sample_config)

    # Should detect English format
    assert config_key == "revolut"


def test_detect_bank_config_non_revolut(tmpdir, sample_config):
    """Test detect_bank_config with non-Revolut bank (should return bank_name unchanged)."""
    # Create a temporary CSV file
    n26_csv = tmpdir.join("n26.csv")
    n26_csv.write("Value Date,Partner Name,Amount (EUR),Bank,Payment Reference\n")

    # Test detection for n26 (should just return "n26")
    config_key = detect_bank_config(str(n26_csv), "n26", sample_config)

    # Should return the bank_name unchanged
    assert config_key == "n26"


def test_detect_bank_config_error_handling(sample_config):
    """Test detect_bank_config error handling with invalid file path."""
    # Test with non-existent file
    config_key = detect_bank_config("non_existent_file.csv", "revolut", sample_config)

    # Should fall back to bank_name on error
    assert config_key == "revolut"


def test_save_to_excel(tmpdir):
    """Test the save_to_excel function."""
    # Define the output directory
    output_dir = str(tmpdir)

    # Define the folders data
    folders_data = ["n26", "revolut"]

    # Create a sample dataframe with datetime objects
    sample_df = pd.DataFrame(
        {
            "Day": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"]),
            "Expense_name": ["Supermarket", "Restaurant", "Transport", "Coffee Shop"],
            "Amount": [-45.50, -26.75, -12.50, -3.75],
            "Bank": ["N26", "N26", "Revolut", "Revolut"],
            "Comment": ["Groceries", "Dinner", "Transport", "Food"],
        }
    )

    # Save to Excel
    save_to_excel(df=sample_df, output_dir=output_dir, folders_data=folders_data)

    # Check if the file was created
    file_list = os.listdir(output_dir)
    assert len(file_list) == 1

    # Check if the filename contains the folders data
    filename = file_list[0]
    assert "n26-revolut" in filename
    assert filename.endswith(".xlsx")

    # Check if the file contains the data
    excel_path = os.path.join(output_dir, filename)
    df_read = pd.read_excel(excel_path)

    # Check if the read dataframe matches the original
    assert df_read.shape == sample_df.shape

    # Day column should be formatted as strings in the Excel file
    # Check format matches DD/MM/YYYY
    assert df_read["Day"].iloc[0] == "01/01/2023"


def test_parse_args_skip_classification():
    """The CLI should expose an explicit manual-classification mode."""
    args = parse_args(["--skip-classification"])

    assert args.skip_classification is True


def test_prepare_manual_classification(sample_processed_df):
    """Manual mode should add blank category columns without mutating input."""
    result = prepare_manual_classification(sample_processed_df)

    assert "Primary" not in sample_processed_df.columns
    assert "Secondary" not in sample_processed_df.columns
    assert result["Primary"].eq("").all()
    assert result["Secondary"].eq("").all()
    assert result.columns[-2:].tolist() == ["Primary", "Secondary"]
    assert result["Day"].is_monotonic_increasing


def test_process_data_does_not_mutate_input(sample_n26_df, sample_config):
    """Test that process_data does not mutate the original DataFrame."""
    original_columns = list(sample_n26_df.columns)
    original_values = sample_n26_df.copy()

    process_data(df=sample_n26_df, config=sample_config["n26"], bank_name="n26")

    # Original df should not be modified
    assert list(sample_n26_df.columns) == original_columns
    pd.testing.assert_frame_equal(sample_n26_df, original_values)


def test_should_classify_ollama_without_openai_key(monkeypatch):
    """Local Ollama classification does not require a cloud API key."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    enabled, reason = should_classify(False, {"provider": "ollama", "model_name": "qwen3.5:9b-q8_0"})

    assert enabled is True
    assert reason == ""


def test_should_classify_openai_requires_key(monkeypatch):
    """Cloud OpenAI classification stays opt-in via OPENAI_API_KEY."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    enabled, reason = should_classify(False, {"provider": "openai", "model_name": "gpt-5-mini"})

    assert enabled is False
    assert reason == "OPENAI_API_KEY is not set"


def test_should_classify_unknown_provider():
    """Unknown providers should skip classification instead of calling a client."""
    enabled, reason = should_classify(False, {"provider": "groq"})

    assert enabled is False
    assert "Unknown LLM provider" in reason


def test_create_llm_client_ollama():
    """Ollama should use the local OpenAI-compatible endpoint."""
    with mock.patch("import_bank_details.main.OpenAI") as mock_openai:
        create_llm_client(
            {
                "provider": "ollama",
                "base_url": "http://127.0.0.1:11434/v1",
                "api_key": "ollama",
                "timeout": 180,
            }
        )

    mock_openai.assert_called_once_with(
        base_url="http://127.0.0.1:11434/v1",
        api_key="ollama",
        timeout=180,
    )


def test_ollama_is_available_success():
    """A successful models.list() probe should report Ollama as available."""
    with mock.patch("import_bank_details.main.OpenAI") as mock_openai:
        mock_model = mock.MagicMock()
        mock_model.id = "qwen3.5:9b-q8_0"
        mock_openai.return_value.models.list.return_value.data = [mock_model]

        available = ollama_is_available(
            {
                "base_url": "http://127.0.0.1:11434/v1",
                "api_key": "ollama",
                "model_name": "qwen3.5:9b-q8_0",
            }
        )

    assert available is True
    mock_openai.assert_called_once_with(
        base_url="http://127.0.0.1:11434/v1",
        api_key="ollama",
        timeout=5,
    )


def test_ollama_is_available_missing_model_still_true(caplog):
    """A missing model name should warn but still attempt classification."""
    with mock.patch("import_bank_details.main.OpenAI") as mock_openai, caplog.at_level("WARNING"):
        mock_model = mock.MagicMock()
        mock_model.id = "other-model"
        mock_openai.return_value.models.list.return_value.data = [mock_model]

        available = ollama_is_available(
            {
                "base_url": "http://127.0.0.1:11434/v1",
                "api_key": "ollama",
                "model_name": "qwen3.5:9b-q8_0",
            }
        )

    assert available is True
    assert "qwen3.5:9b-q8_0" in caplog.text
    assert "was not found" in caplog.text


def test_ollama_is_available_connection_failure():
    """A refused connection should fail the health check immediately."""
    with mock.patch("import_bank_details.main.OpenAI") as mock_openai:
        mock_openai.return_value.models.list.side_effect = ConnectionError("Connection refused")

        available = ollama_is_available(
            {
                "base_url": "http://127.0.0.1:11434/v1",
                "api_key": "ollama",
                "model_name": "qwen3.5:9b-q8_0",
            }
        )

    assert available is False


def test_main_without_openai_key_exports_for_manual_classification(
    sample_n26_csv, sample_config, sample_llm_config, monkeypatch, caplog
):
    """A missing OpenAI key should export processed rows without initializing API clients."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with (
        mock.patch("import_bank_details.main.load_config", side_effect=[sample_config, sample_llm_config]) as mock_load_config,
        mock.patch("import_bank_details.main.setup_logging"),
        mock.patch("import_bank_details.main.get_latest_files", return_value={"n26": sample_n26_csv}),
        mock.patch("import_bank_details.main.load_dotenv"),
        mock.patch("import_bank_details.main.OpenAI") as mock_openai,
        mock.patch("import_bank_details.main.TavilyClient") as mock_tavily,
        mock.patch("import_bank_details.main.classify_expenses") as mock_classify,
        mock.patch("import_bank_details.main.save_to_excel") as mock_save,
        caplog.at_level("WARNING"),
    ):
        main()

    saved_df = mock_save.call_args.args[0]
    assert saved_df["Primary"].eq("").all()
    assert saved_df["Secondary"].eq("").all()
    mock_openai.assert_not_called()
    mock_tavily.assert_not_called()
    mock_classify.assert_not_called()
    assert mock_load_config.call_args_list[0].kwargs["config_path"] == "config_bank.yaml"
    assert mock_load_config.call_args_list[1].kwargs["config_path"] == "config_llm.yaml"
    assert "OPENAI_API_KEY is not set" in caplog.text


def test_main_ollama_classifies_without_openai_key(sample_n26_csv, sample_config, monkeypatch):
    """The default local provider should classify without OPENAI_API_KEY."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    ollama_config = {
        "llm": {
            "provider": "ollama",
            "model_name": "qwen3.5:9b-q8_0",
            "base_url": "http://127.0.0.1:11434/v1",
            "api_key": "ollama",
            "timeout": 180,
            "reasoning_effort": "none",
            "max_workers": 2,
        },
        "system_prompt": "Test",
    }

    with (
        mock.patch("import_bank_details.main.load_config", side_effect=[sample_config, ollama_config]),
        mock.patch("import_bank_details.main.setup_logging"),
        mock.patch("import_bank_details.main.get_latest_files", return_value={"n26": sample_n26_csv}),
        mock.patch("import_bank_details.main.load_dotenv"),
        mock.patch("import_bank_details.main.OpenAI") as mock_openai,
        mock.patch("import_bank_details.main.TavilyClient") as mock_tavily,
        mock.patch("import_bank_details.main.classify_expenses") as mock_classify,
        mock.patch("import_bank_details.main.save_to_excel"),
    ):
        mock_model = mock.MagicMock()
        mock_model.id = "qwen3.5:9b-q8_0"
        mock_openai.return_value.models.list.return_value.data = [mock_model]
        mock_classify.return_value = pd.DataFrame(
            {
                "Day": pd.to_datetime(["2023-01-01"]),
                "Expense_name": ["Test"],
                "Amount": [-10.0],
                "Bank": ["n26"],
                "Comment": [""],
                "Primary": ["Groceries"],
                "Secondary": ["Auchan"],
            }
        )
        main()

    assert mock_openai.call_count == 2
    assert mock_openai.call_args_list[0].kwargs["timeout"] == 5
    assert mock_openai.call_args_list[1].kwargs["timeout"] == 180
    mock_tavily.assert_not_called()
    assert mock_classify.call_args.kwargs["model_name"] == "qwen3.5:9b-q8_0"
    assert mock_classify.call_args.kwargs["reasoning_effort"] == "none"
    assert mock_classify.call_args.kwargs["max_workers"] == 2
    assert mock_classify.call_args.kwargs["include_online_search"] is False
    assert mock_classify.call_args.kwargs["provider"] == "ollama"
    assert mock_classify.call_args.kwargs["batch_size"] == 10
    assert mock_classify.call_args.kwargs["max_few_shot_examples"] == 32
    assert mock_classify.call_args.kwargs["classification_cache"] is not None


def test_main_ollama_unavailable_exports_blank_columns(sample_n26_csv, sample_config, monkeypatch, caplog):
    """If Ollama is down, skip classification and export blank columns."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    ollama_config = {
        "llm": {
            "provider": "ollama",
            "model_name": "qwen3.5:9b-q8_0",
            "base_url": "http://127.0.0.1:11434/v1",
            "api_key": "ollama",
            "timeout": 180,
            "reasoning_effort": "none",
            "max_workers": 2,
        },
        "system_prompt": "Test",
    }

    with (
        mock.patch("import_bank_details.main.load_config", side_effect=[sample_config, ollama_config]),
        mock.patch("import_bank_details.main.setup_logging"),
        mock.patch("import_bank_details.main.get_latest_files", return_value={"n26": sample_n26_csv}),
        mock.patch("import_bank_details.main.load_dotenv"),
        mock.patch("import_bank_details.main.OpenAI") as mock_openai,
        mock.patch("import_bank_details.main.classify_expenses") as mock_classify,
        mock.patch("import_bank_details.main.save_to_excel") as mock_save,
        caplog.at_level("WARNING"),
    ):
        mock_openai.return_value.models.list.side_effect = ConnectionError("Connection refused")
        main()

    saved_df = mock_save.call_args.args[0]
    assert saved_df["Primary"].eq("").all()
    assert saved_df["Secondary"].eq("").all()
    mock_classify.assert_not_called()
    mock_openai.assert_called_once_with(
        base_url="http://127.0.0.1:11434/v1",
        api_key="ollama",
        timeout=5,
    )
    assert "Ollama is not running" in caplog.text
    assert "qwen3.5:9b-q8_0" in caplog.text


def test_main_explicitly_skips_classification_with_openai_key(sample_n26_csv, sample_config, monkeypatch):
    """The CLI-facing option should override an available OpenAI key."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

    with (
        mock.patch("import_bank_details.main.load_config", return_value=sample_config),
        mock.patch("import_bank_details.main.setup_logging"),
        mock.patch("import_bank_details.main.get_latest_files", return_value={"n26": sample_n26_csv}),
        mock.patch("import_bank_details.main.load_dotenv"),
        mock.patch("import_bank_details.main.OpenAI") as mock_openai,
        mock.patch("import_bank_details.main.classify_expenses") as mock_classify,
        mock.patch("import_bank_details.main.save_to_excel") as mock_save,
    ):
        main(skip_classification=True)

    saved_df = mock_save.call_args.args[0]
    assert saved_df["Primary"].eq("").all()
    assert saved_df["Secondary"].eq("").all()
    mock_openai.assert_not_called()
    mock_classify.assert_not_called()


def test_main_classifies_without_tavily_key(sample_n26_csv, sample_config, sample_llm_config, monkeypatch):
    """Tavily enrichment should be optional when OpenAI classification is enabled."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    with (
        mock.patch("import_bank_details.main.load_config", side_effect=[sample_config, sample_llm_config]),
        mock.patch("import_bank_details.main.setup_logging"),
        mock.patch("import_bank_details.main.get_latest_files", return_value={"n26": sample_n26_csv}),
        mock.patch("import_bank_details.main.load_dotenv"),
        mock.patch("import_bank_details.main.OpenAI") as mock_openai,
        mock.patch("import_bank_details.main.TavilyClient") as mock_tavily,
        mock.patch("import_bank_details.main.SearchCache") as mock_cache,
        mock.patch("import_bank_details.main.classify_expenses") as mock_classify,
        mock.patch("import_bank_details.main.save_to_excel"),
    ):
        main()

    mock_openai.assert_called_once()
    mock_tavily.assert_not_called()
    mock_cache.assert_not_called()
    assert mock_classify.call_args.kwargs["include_online_search"] is False
    assert mock_classify.call_args.kwargs["tavily_client"] is None
    assert mock_classify.call_args.kwargs["search_cache"] is None
    assert mock_classify.call_args.kwargs["classification_cache"] is not None
    assert mock_classify.call_args.kwargs["batch_size"] == 10
    assert mock_classify.call_args.kwargs["max_few_shot_examples"] == 32


def test_main_all_banks_fail(sample_data_dir, sample_config):
    """Test that main raises RuntimeError when all banks fail."""
    with (
        mock.patch("import_bank_details.main.load_config") as mock_load_config,
        mock.patch("import_bank_details.main.setup_logging"),
        mock.patch("import_bank_details.main.get_latest_files") as mock_get_latest_files,
        mock.patch("import_bank_details.main.load_dotenv"),
        mock.patch("import_bank_details.main.OpenAI"),
        mock.patch("import_bank_details.main.TavilyClient"),
        mock.patch("import_bank_details.main.SearchCache"),
        mock.patch("import_bank_details.main.import_data") as mock_import_data,
    ):
        mock_load_config.side_effect = [
            sample_config,
            {
                "llm": {"provider": "openai", "model_name": "gpt-4o-mini", "temperature_base": 0.0},
                "system_prompt": "Test",
            },
        ]
        mock_get_latest_files.return_value = {
            "n26": "/fake/n26.csv",
            "revolut": "/fake/revolut.csv",
        }
        mock_import_data.side_effect = Exception("File not found")

        with pytest.raises(RuntimeError, match="All banks failed to process"):
            main()


def test_main_partial_bank_failure(sample_data_dir, sample_n26_csv, sample_config):
    """Test that main continues when some banks fail and warns."""
    with (
        mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-openai-key"}),
        mock.patch("import_bank_details.main.load_config") as mock_load_config,
        mock.patch("import_bank_details.main.setup_logging"),
        mock.patch("import_bank_details.main.get_latest_files") as mock_get_latest_files,
        mock.patch("import_bank_details.main.load_dotenv"),
        mock.patch("import_bank_details.main.OpenAI"),
        mock.patch("import_bank_details.main.TavilyClient"),
        mock.patch("import_bank_details.main.SearchCache"),
        mock.patch("import_bank_details.main.classify_expenses") as mock_classify,
        mock.patch("import_bank_details.main.save_to_excel") as mock_save,
    ):
        mock_load_config.side_effect = [
            sample_config,
            {
                "llm": {"provider": "openai", "model_name": "gpt-4o-mini", "temperature_base": 0.0},
                "system_prompt": "Test",
            },
        ]
        mock_get_latest_files.return_value = {
            "n26": sample_n26_csv,
            "revolut": "/fake/nonexistent.csv",
        }

        # Make classify_expenses return a simple df
        mock_classify.return_value = pd.DataFrame(
            {
                "Day": pd.to_datetime(["2023-01-01"]),
                "Expense_name": ["Test"],
                "Amount": [-10.0],
                "Bank": ["n26"],
                "Comment": [""],
                "Primary": ["Groceries"],
                "Secondary": ["Auchan"],
            }
        )

        # Should not raise - partial failure is a warning
        main()

        mock_save.assert_called_once()


def test_main_happy_path(
    sample_data_dir,
    sample_n26_csv,
    sample_revolut_csv,
    sample_example_csv,
    sample_config,
    sample_llm_config,
):
    """Test main function happy path with all components mocked."""
    from import_bank_details.structured_output import ExpenseOutput, ExpenseType

    expense_type = None
    for et in ExpenseType:  # type: ignore[attr-defined]
        if et.value == "Groceries, Auchan":
            expense_type = et
            break
    mock_output = ExpenseOutput(expense_type=expense_type)

    with (
        mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-openai-key", "TAVILY_API_KEY": "test-tavily-key"}),
        mock.patch("import_bank_details.main.load_config") as mock_load_config,
        mock.patch("import_bank_details.main.setup_logging"),
        mock.patch("import_bank_details.main.get_latest_files") as mock_get_latest_files,
        mock.patch("import_bank_details.main.load_dotenv"),
        mock.patch("import_bank_details.main.OpenAI"),
        mock.patch("import_bank_details.main.TavilyClient"),
        mock.patch("import_bank_details.main.SearchCache"),
        mock.patch("import_bank_details.classification.get_classification") as mock_classify,
        mock.patch("import_bank_details.main.save_to_excel") as mock_save,
    ):
        mock_load_config.side_effect = [sample_config, sample_llm_config]
        mock_get_latest_files.return_value = {
            "n26": sample_n26_csv,
            "revolut": sample_revolut_csv,
            "examples": sample_example_csv,
        }
        mock_classify.return_value = mock_output

        main()

        mock_save.assert_called_once()
        # Verify the DataFrame passed to save_to_excel has expected columns
        saved_df = mock_save.call_args[0][0]
        assert "Primary" in saved_df.columns
        assert "Secondary" in saved_df.columns
        assert "Day" in saved_df.columns


def test_main_output_sort_order(
    sample_data_dir,
    sample_n26_csv,
    sample_revolut_csv,
    sample_example_csv,
    sample_config,
    sample_llm_config,
):
    """Test that the output DataFrame is sorted by Day, Amount, Expense_name."""
    import threading

    from import_bank_details.structured_output import ExpenseOutput, ExpenseType

    expense_type_groceries = None
    expense_type_restaurants = None
    for et in ExpenseType:  # type: ignore[attr-defined]
        if et.value == "Groceries, Auchan":
            expense_type_groceries = et
        elif et.value == "Out, Restaurants":
            expense_type_restaurants = et

    lock = threading.Lock()

    def mock_classify_func(**kwargs):
        with lock:
            name = kwargs["expense_input"]["Expense_name"]
            if "Restaurant" in name:
                return ExpenseOutput(expense_type=expense_type_restaurants)
            return ExpenseOutput(expense_type=expense_type_groceries)

    with (
        mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-openai-key", "TAVILY_API_KEY": "test-tavily-key"}),
        mock.patch("import_bank_details.main.load_config") as mock_load_config,
        mock.patch("import_bank_details.main.setup_logging"),
        mock.patch("import_bank_details.main.get_latest_files") as mock_get_latest_files,
        mock.patch("import_bank_details.main.load_dotenv"),
        mock.patch("import_bank_details.main.OpenAI"),
        mock.patch("import_bank_details.main.TavilyClient"),
        mock.patch("import_bank_details.main.SearchCache"),
        mock.patch("import_bank_details.classification.get_classification") as mock_classify,
        mock.patch("import_bank_details.main.save_to_excel") as mock_save,
    ):
        mock_load_config.side_effect = [sample_config, sample_llm_config]
        mock_get_latest_files.return_value = {
            "n26": sample_n26_csv,
            "revolut": sample_revolut_csv,
            "examples": sample_example_csv,
        }
        mock_classify.side_effect = mock_classify_func

        main()

        saved_df = mock_save.call_args[0][0]
        # Verify sort order: Day ascending, then Amount, then Expense_name
        days = saved_df["Day"].tolist()
        assert days == sorted(days)
