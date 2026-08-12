"""Tests for the classification cache and expense-name cleaner."""

import json
import logging

from import_bank_details.classification_cache import ClassificationCache
from import_bank_details.expense_names import clean_expense_name


class TestCleanExpenseName:
    """Tests for clean_expense_name."""

    def test_strips_known_prefixes(self):
        assert clean_expense_name("PAYPAL *Lidl") == "Lidl"
        assert clean_expense_name("SumUp  *Coffee") == "Coffee"
        assert clean_expense_name("LSP*Store") == "Store"
        assert clean_expense_name("CRV*Shop") == "Shop"
        assert clean_expense_name("PAY.nl*Merchant") == "Merchant"
        assert clean_expense_name("UZR*Place") == "Place"
        assert clean_expense_name("luca Bakery") == "Bakery"

    def test_strips_whitespace(self):
        assert clean_expense_name("  Lidl  ") == "Lidl"

    def test_empty_after_cleaning(self):
        assert clean_expense_name("PAYPAL *") == ""
        assert clean_expense_name("SumUp  *") == ""
        assert clean_expense_name("   ") == ""


class TestClassificationCache:
    """Tests for ClassificationCache."""

    def test_init(self):
        cache = ClassificationCache()
        assert cache._cache == {}
        assert cache._loaded is False

    def test_two_instances_are_independent(self):
        cache1 = ClassificationCache()
        cache2 = ClassificationCache()
        cache1._cache["Lidl"] = {"Primary": "Groceries", "Secondary": "Lidl"}
        cache1._loaded = True
        assert cache2._cache == {}
        assert cache2._loaded is False

    def test_get_cache_path_default(self):
        cache = ClassificationCache()
        default_path = cache.get_cache_path()
        assert str(default_path).endswith("data/examples/classification_cache.json")

    def test_get_cache_path_custom(self, tmp_path):
        cache = ClassificationCache()
        custom_path = cache.get_cache_path(tmp_path)
        assert custom_path == tmp_path / "classification_cache.json"
        assert tmp_path.exists()

    def test_put_and_get_roundtrip(self, tmp_path):
        cache = ClassificationCache()
        cache.put("Lidl", "Groceries", "Lidl", tmp_path)
        assert cache.get("Lidl", tmp_path) == {"Primary": "Groceries", "Secondary": "Lidl"}

    def test_put_persists_to_disk(self, tmp_path):
        cache = ClassificationCache()
        cache.put("Lidl", "Groceries", "Lidl", tmp_path)

        cache_file = tmp_path / "classification_cache.json"
        assert cache_file.exists()
        with open(cache_file) as f:
            data = json.load(f)
        assert data == {"Lidl": {"Primary": "Groceries", "Secondary": "Lidl"}}

    def test_name_cleaning_shares_key(self, tmp_path):
        cache = ClassificationCache()
        cache.put("PAYPAL *Lidl", "Groceries", "Lidl", tmp_path)
        assert cache.get("Lidl", tmp_path) == {"Primary": "Groceries", "Secondary": "Lidl"}
        assert cache.get("PAYPAL *Lidl", tmp_path) == {"Primary": "Groceries", "Secondary": "Lidl"}

        cache_file = tmp_path / "classification_cache.json"
        with open(cache_file) as f:
            data = json.load(f)
        assert list(data.keys()) == ["Lidl"]

    def test_empty_names_not_stored(self, tmp_path):
        cache = ClassificationCache()
        cache.put("", "Groceries", "Lidl", tmp_path)
        cache.put("PAYPAL *", "Groceries", "Lidl", tmp_path)
        cache.put("   ", "Groceries", "Lidl", tmp_path)

        assert cache.get("", tmp_path) is None
        assert cache.get("PAYPAL *", tmp_path) is None
        assert not (tmp_path / "classification_cache.json").exists()

    def test_empty_primary_or_secondary_not_stored(self, tmp_path):
        cache = ClassificationCache()
        cache.put("Lidl", "", "Lidl", tmp_path)
        cache.put("Lidl", "Groceries", "", tmp_path)

        assert cache.get("Lidl", tmp_path) is None
        assert not (tmp_path / "classification_cache.json").exists()

    def test_get_returns_none_for_missing_key(self, tmp_path):
        cache = ClassificationCache()
        assert cache.get("missing", tmp_path) is None

    def test_get_returns_none_for_empty_cleaned_name(self, tmp_path):
        cache = ClassificationCache()
        cache.put("Lidl", "Groceries", "Lidl", tmp_path)
        assert cache.get("PAYPAL *", tmp_path) is None
        assert cache.get("", tmp_path) is None

    def test_lazy_load_from_disk(self, tmp_path):
        cache_file = tmp_path / "classification_cache.json"
        with open(cache_file, "w") as f:
            json.dump({"Lidl": {"Primary": "Groceries", "Secondary": "Lidl"}}, f)

        cache = ClassificationCache()
        assert cache.get("Lidl", tmp_path) == {"Primary": "Groceries", "Secondary": "Lidl"}

    def test_corrupted_cache_file_recovers(self, tmp_path, caplog):
        cache_file = tmp_path / "classification_cache.json"
        cache_file.write_text("invalid json{{{")

        cache = ClassificationCache()
        with caplog.at_level(logging.WARNING):
            assert cache.get("any_key", tmp_path) is None
        assert "corrupted" in caplog.text.lower()

        cache.put("Lidl", "Groceries", "Lidl", tmp_path)
        assert cache.get("Lidl", tmp_path) == {"Primary": "Groceries", "Secondary": "Lidl"}

    def test_custom_cache_path(self, tmp_path):
        cache = ClassificationCache()
        cache.put("Lidl", "Groceries", "Lidl", tmp_path)
        assert (tmp_path / "classification_cache.json").exists()
        assert cache.get("Lidl", tmp_path) == {"Primary": "Groceries", "Secondary": "Lidl"}
