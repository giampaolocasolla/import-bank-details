"""Tests for the structured_output module."""

import pytest
from pydantic import ValidationError

from import_bank_details.structured_output import ExpenseEntry, ExpenseInput, ExpenseOutput, ExpenseType


def test_expense_type_enum():
    """Test the ExpenseType enum creation and structure."""
    # Check that the enum was created correctly
    assert hasattr(ExpenseType, "RENT")
    assert hasattr(ExpenseType, "RESTAURANTS")

    # Check some enum values
    assert ExpenseType.RENT.value == "Housing, Rent"
    assert ExpenseType.RESTAURANTS.value == "Out, Restaurants"


def test_expense_output_properties():
    """Test the properties of the ExpenseOutput model."""
    # Create an ExpenseOutput instance
    expense_output = ExpenseOutput(expense_type=ExpenseType.RESTAURANTS)

    # Test the category and subcategory properties
    assert expense_output.category == "Out"
    assert expense_output.subcategory == "Restaurants"


def test_expense_output_validation():
    """Test validation in the ExpenseOutput model."""
    # Test with invalid expense type
    with pytest.raises(ValidationError):
        ExpenseOutput(expense_type="Not an enum")


def test_expense_input_model():
    """Test the ExpenseInput model."""
    # Create a valid ExpenseInput
    expense_input = ExpenseInput(
        Day="01/01/2023", Expense_name="Restaurant Dinner", Amount="45.50", Bank="Revolut", Comment="Family dinner"
    )

    # Check the fields
    assert expense_input.Day == "01/01/2023"
    assert expense_input.Expense_name == "Restaurant Dinner"
    assert expense_input.Amount == "45.50"
    assert expense_input.Bank == "Revolut"
    assert expense_input.Comment == "Family dinner"

    # Test model_dump method
    data = expense_input.model_dump()
    assert "Day" in data
    assert "Expense_name" in data
    assert "Amount" in data
    assert "Bank" in data
    assert "Comment" in data


def test_expense_entry_model():
    """Test the ExpenseEntry model."""
    # Create an ExpenseInput
    expense_input = ExpenseInput(
        Day="01/01/2023", Expense_name="Restaurant Dinner", Amount="45.50", Bank="Revolut", Comment="Family dinner"
    )

    # Create an ExpenseOutput
    expense_output = ExpenseOutput(expense_type=ExpenseType.RESTAURANTS)

    # Create an ExpenseEntry with both input and output
    expense_entry = ExpenseEntry(input=expense_input, output=expense_output)

    # Check both input and output
    assert expense_entry.input == expense_input
    assert expense_entry.output == expense_output

    # Test with only input (output is optional)
    expense_entry_no_output = ExpenseEntry(input=expense_input)
    assert expense_entry_no_output.input == expense_input
    assert expense_entry_no_output.output is None
