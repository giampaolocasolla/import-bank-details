"""Tests for KATE-style few-shot example retrieval."""

from typing import Any

from import_bank_details.example_retriever import ExampleRetriever


def _example(name: str, output: str, comment: str = "") -> dict[str, Any]:
    return {
        "input": {
            "Day": "01/01/2023",
            "Expense_name": name,
            "Amount": "10.00",
            "Bank": "N26",
            "Comment": comment,
        },
        "output": output,
    }


def _query(name: str, comment: str = "") -> dict[str, str]:
    return {
        "Day": "02/01/2023",
        "Expense_name": name,
        "Amount": "12.00",
        "Bank": "N26",
        "Comment": comment,
    }


def _names(examples: list[dict[str, Any]]) -> list[str]:
    return [example["input"]["Expense_name"] for example in examples]


def _filler(count: int, prefix: str = "UnrelatedShop") -> list[dict[str, Any]]:
    return [_example(f"{prefix}{i}", "Out, Restaurants") for i in range(count)]


def test_small_pool_returns_all_in_original_order() -> None:
    examples = [
        _example("Lidl", "Groceries, Lidl"),
        _example("Rewe", "Groceries, Rewe"),
        _example("Aldi", "Groceries, Aldi"),
    ]
    retriever = ExampleRetriever(examples, max_examples=32)
    result = retriever.retrieve_for_batch([_query("SomethingElse")])
    assert _names(result) == ["Lidl", "Rewe", "Aldi"]
    assert result == examples


def test_empty_pool_returns_empty() -> None:
    retriever = ExampleRetriever([], max_examples=32)
    assert retriever.retrieve_for_batch([_query("REWE")]) == []


def test_max_examples_zero_returns_empty() -> None:
    examples = [_example("Lidl", "Groceries, Lidl"), *_filler(5)]
    retriever = ExampleRetriever(examples, max_examples=0)
    assert retriever.retrieve_for_batch([_query("Lidl")]) == []


def test_negative_max_examples_clamps_to_zero() -> None:
    examples = [_example("Lidl", "Groceries, Lidl"), *_filler(5)]
    retriever = ExampleRetriever(examples, max_examples=-3)
    assert retriever.retrieve_for_batch([_query("Lidl")]) == []


def test_brand_specificity_ranks_rewe_above_lidl_and_aldi() -> None:
    examples = [
        _example("Lidl SAGT DANKE BERLIN", "Groceries, Lidl"),
        _example("REWE SAGT DANKE KOELN", "Groceries, Rewe"),
        _example("Aldi Sued Muenchen", "Groceries, Aldi"),
        *_filler(10),
    ]
    retriever = ExampleRetriever(examples, max_examples=1)
    result = retriever.retrieve_for_batch([_query("REWE SAGT DANKE MUENCHEN")])
    assert _names(result) == ["REWE SAGT DANKE KOELN"]


def test_prefix_stripping_matches_spotify_example() -> None:
    examples = [
        _example("SPOTIFY", "Leisure, OtherLeisure"),
        _example("Lidl", "Groceries, Lidl"),
        *_filler(10),
    ]
    retriever = ExampleRetriever(examples, max_examples=1)
    result = retriever.retrieve_for_batch([_query("PAYPAL *SPOTIFY")])
    assert _names(result) == ["SPOTIFY"]


def test_abbreviation_ranks_amazon_above_restaurant() -> None:
    examples = [
        _example("AMAZON", "Groceries, Amazon"),
        _example("Vapiano Restaurant", "Out, Restaurants"),
        *_filler(10),
    ]
    retriever = ExampleRetriever(examples, max_examples=1)
    result = retriever.retrieve_for_batch([_query("AMZN MKTP")])
    assert _names(result) == ["AMAZON"]


def test_batch_merge_keeps_examples_for_both_merchants() -> None:
    examples = [
        _example("REWE SAGT DANKE", "Groceries, Rewe"),
        _example("Lidl SAGT DANKE", "Groceries, Lidl"),
        _example("Vapiano", "Out, Restaurants"),
        _example("Aldi", "Groceries, Aldi"),
        *_filler(10),
    ]
    retriever = ExampleRetriever(examples, max_examples=2)
    result = retriever.retrieve_for_batch(
        [
            _query("REWE MUENCHEN"),
            _query("Vapiano Berlin"),
        ]
    )
    names = set(_names(result))
    assert "REWE SAGT DANKE" in names
    assert "Vapiano" in names
    assert len(result) == 2


def test_cap_returns_exactly_max_examples() -> None:
    examples = [_example(f"REWE {i}", "Groceries, Rewe") for i in range(200)]
    retriever = ExampleRetriever(examples, max_examples=32)
    result = retriever.retrieve_for_batch([_query("REWE 0")])
    assert len(result) == 32


def test_max_examples_above_hard_cap_clamps_to_100() -> None:
    examples = [_example(f"REWE {i}", "Groceries, Rewe") for i in range(200)]
    retriever = ExampleRetriever(examples, max_examples=150)
    result = retriever.retrieve_for_batch([_query("REWE 0")])
    assert len(result) == 100


def test_ordering_closest_example_is_last() -> None:
    examples = [
        _example("Lidl Berlin", "Groceries, Lidl"),
        _example("REWE SAGT DANKE MUENCHEN", "Groceries, Rewe"),
        _example("Aldi Nord", "Groceries, Aldi"),
        *_filler(5),
    ]
    retriever = ExampleRetriever(examples, max_examples=3)
    result = retriever.retrieve_for_batch([_query("REWE SAGT DANKE MUENCHEN")])
    assert result[-1]["input"]["Expense_name"] == "REWE SAGT DANKE MUENCHEN"


def test_zero_overlap_returns_empty_when_pool_exceeds_max() -> None:
    examples = [_example(f"Lidl {i}", "Groceries, Lidl") for i in range(40)]
    retriever = ExampleRetriever(examples, max_examples=32)
    result = retriever.retrieve_for_batch([_query("QQQQQQQQ")])
    assert result == []


def test_empty_merchant_names_with_large_pool_return_empty() -> None:
    examples = [_example("", "Groceries, OtherGroceries") for _ in range(40)]
    retriever = ExampleRetriever(examples, max_examples=32)
    assert retriever.retrieve_for_batch([_query("REWE")]) == []


def test_comment_is_included_in_retrieval_text() -> None:
    examples = [
        _example("Store", "Groceries, OtherGroceries", comment="REWE MUENCHEN"),
        _example("Store", "Out, Restaurants", comment="Vapiano"),
        *_filler(10),
    ]
    retriever = ExampleRetriever(examples, max_examples=1)
    result = retriever.retrieve_for_batch([_query("Store", comment="REWE")])
    assert result[0]["output"] == "Groceries, OtherGroceries"
