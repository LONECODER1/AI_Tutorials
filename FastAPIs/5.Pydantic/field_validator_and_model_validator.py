from typing import Dict, List, Optional
from pydantic import BaseModel, EmailStr, Field, ValidationError, field_validator, model_validator


class Patient(BaseModel):
    name: str = Field(..., description="Name of the patient")
    email: EmailStr = Field(..., description="Email of the patient")
    age: int = Field(..., description="Age of the patient")
    weight: float = Field(..., description="Weight of the patient in kg")
    married: bool = Field(default=False, description="Marital status")
    allergies: List[str] = Field(default_factory=list, description="List of allergies")
    contact_details: Dict[str, str] = Field(..., description="Contact details")

    # ==========================================
    # FIELD VALIDATORS (Single-field validation)
    # ==========================================

    # 1. Transformation Validator: transforms name to uppercase
    @field_validator("name")
    @classmethod
    def transform_name(cls, value: str) -> str:
        return value.upper()

    # 2. Custom Business Logic Validator: checks allowed email domains
    @field_validator("email")
    @classmethod
    def email_validator(cls, value: str) -> str:
        valid_domains = ["icici.com", "hdfc.com", "gmail.com"]
        domain = value.split("@")[-1]
        if domain not in valid_domains:
            raise ValueError(f"Email domain must be one of: {valid_domains}")
        return value

    # 3. Range Validator: validates that age is realistic
    @field_validator("age")
    @classmethod
    def validate_age(cls, value: int) -> int:
        if value < 0 or value > 120:
            raise ValueError("Age must be between 0 and 120")
        return value

    # ==========================================
    # MODEL VALIDATOR (Cross-field validation)
    # ==========================================

    # 4. Model Validator: validates relationships between multiple fields
    @model_validator(mode="after")
    def validate_marriage_age(self) -> "Patient":
        # Cross-field check: a patient cannot be married if they are under 18
        if self.married and self.age < 18:
            raise ValueError("A patient cannot be married if they are under 18 years old.")
        return self


def update_patient_data(patient: Patient):
    print("Name (transformed):", patient.name)
    print("Age (coerced int):", patient.age)
    print("Email (validated):", patient.email)
    print("Allergies:", patient.allergies)
    print("Married:", patient.married)
    print("updated successfully!\n")


# ----------------------------------------------------
# 1. Valid Example: Passes field and model validators
# ----------------------------------------------------
print("--- Test 1: Valid Patient ---")
patient_info = {
    "name": "nitish",
    "email": "abc@icici.com",
    "age": "30",  # string gets coerced to int
    "weight": 75.2,
    "married": True,
    "allergies": ["pollen", "dust"],
    "contact_details": {"phone": "2353462"},
}

patient1 = Patient(**patient_info)
update_patient_data(patient1)


# ----------------------------------------------------
# 2. Invalid Example: Triggers the Model Validator
# ----------------------------------------------------
print("--- Test 2: Triggering Model Validator (married=True, age=16) ---")
try:
    invalid_patient_info = {
        "name": "rohit",
        "email": "rohit@gmail.com",
        "age": 16,
        "weight": 55.0,
        "married": True,  # invalid because age < 18
        "allergies": [],
        "contact_details": {"phone": "9876543210"},
    }
    patient2 = Patient(**invalid_patient_info)
except ValidationError as e:
    print("Caught expected ValidationError from model_validator:")
    print(e)
