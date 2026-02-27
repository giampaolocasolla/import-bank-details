import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, field_validator

logger = logging.getLogger(__name__)

_DEFAULT_CATEGORIES_PATH = Path(__file__).parent / "categories.yaml"


def load_expense_type_enum(categories_path: Optional[Path] = None) -> type:
    """Load expense categories from YAML and create the ExpenseType enum.

    Args:
        categories_path: Path to the categories YAML file.
            Defaults to the co-located categories.yaml.

    Returns:
        A dynamically created Enum class.
    """
    path = categories_path or _DEFAULT_CATEGORIES_PATH
    with open(path, "r", encoding="utf-8") as f:
        categories: Dict[str, Any] = yaml.safe_load(f)

    enum_members: Dict[str, str] = {}
    for category, subcategories in categories.items():
        for subcategory in subcategories:
            enum_member_name = subcategory.upper().replace(" ", "_")
            if enum_member_name in enum_members:
                enum_member_name = f"{category.upper()}_{enum_member_name}"
            enum_members[enum_member_name] = f"{category}, {subcategory}"

    return Enum("ExpenseType", enum_members)  # type: ignore


ExpenseType = load_expense_type_enum()


class ExpenseOutput(BaseModel):
    """The type of the expense, categorized into main category and subcategory."""

    expense_type: ExpenseType  # type: ignore

    @property
    def category(self) -> str:
        return self.expense_type.value.split(", ")[0]  # type: ignore[attr-defined, no-any-return]

    @property
    def subcategory(self) -> str:
        return self.expense_type.value.split(", ")[1]  # type: ignore[attr-defined, no-any-return]

    @field_validator("expense_type")
    @classmethod
    def validate_expense_type(cls, v: Any) -> Any:
        if not isinstance(v, ExpenseType):
            raise ValueError("Invalid expense type")
        return v


class ExpenseInput(BaseModel):
    Day: str
    Expense_name: str
    Amount: str
    Bank: str
    Comment: str


class ExpenseEntry(BaseModel):
    input: ExpenseInput
    output: Optional[ExpenseOutput] = None
