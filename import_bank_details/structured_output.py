import logging
from enum import Enum
from typing import Optional

from pydantic import BaseModel, field_validator

# Get the logger instance
logger = logging.getLogger(__name__)

# Your list of categories and subcategories
data = """
Housing\tRent
Housing\tCondoFee
Housing\tCondoAssurance
Housing\tInternet
Housing\tFurniture
Housing\tMove
Housing\tCleaning
Housing\tHousingTaxes
Housing\tTV
Housing\tElectricity
Housing\tPhone
Transport\tFuel
Transport\tParking
Transport\tRevision
Transport\tTaxi
Transport\tRepair
Transport\tTyres
Transport\tCarTaxes
Transport\tCarAssurance
Transport\tBike
Transport\tTolls
Transport\tPublicTransport
Groceries\tAuchan
Groceries\tDelhaize
Groceries\tLidl
Groceries\tAldi
Groceries\tRewe
Groceries\tKaufland
Groceries\tAmazon
Groceries\tIkeaGroceries
Groceries\tBarber
Groceries\tOtherGroceries
Travel\tNaples
Travel\tSmallTrip
Travel\tLongTrip
Out\tRestaurants
Out\tBar
Out\tTakeAway
Out\tFoodDelivery
Out\tTip
Out\tLeisure
Leisure\tBooks
Leisure\tLeisure
Leisure\tLearning
Leisure\tGames
Leisure\tTech
Leisure\tOtherLeisure
Health\tInsurance
Health\tDoctors
Health\tGlasses
Health\tMedicines
Health\tSport
Health\tSafety
Health\tExams
Gifts\tBirthdays
Gifts\tChristmas
Gifts\tWeddings
Gifts\tDonations
Gifts\tFamily
Gifts\tOtherGifts
Clothing\tMrClothing
Clothing\tOtherClothing
Fees\tBrokers
Fees\tConsulting
Fees\tPostal
Fees\tBanks
Fees\tFines
Fees\tAssurance
Fees\tTax
Fees\tOtherFees
OtherExpenses\tOtherExpenses
"""

# Process the data to create enum members
lines = data.strip().split("\n")
enum_members = {}

for line in lines:
    category, subcategory = line.strip().split("\t")
    enum_member_name = subcategory.upper().replace(" ", "_")
    # Avoid name clashes
    if enum_member_name in enum_members:
        enum_member_name = f"{category.upper()}_{enum_member_name}"
    enum_members[enum_member_name] = f"{category}, {subcategory}"

# Create the ExpenseType enum
ExpenseType = Enum("ExpenseType", enum_members)  # type: ignore


class ExpenseOutput(BaseModel):
    """The type of the expense, categorized into main category and subcategory."""

    expense_type: ExpenseType  # type: ignore

    @property
    def category(self):
        return self.expense_type.value.split(", ")[0]

    @property
    def subcategory(self):
        return self.expense_type.value.split(", ")[1]

    @field_validator("expense_type")
    def validate_expense_type(cls, v):
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
