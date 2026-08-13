"""Helpers for normalizing expense / merchant names."""

PREFIXES_TO_STRIP = ["SumUp  *", "PAYPAL *", "LSP*", "CRV*", "PAY.nl*", "UZR*", "luca "]


def clean_expense_name(expense_name: str) -> str:
    """Strip known payment-processor prefixes and surrounding whitespace."""
    cleaned_name = expense_name
    for text in PREFIXES_TO_STRIP:
        cleaned_name = cleaned_name.replace(text, "")
    return cleaned_name.strip()
