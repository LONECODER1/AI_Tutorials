"""
Computed Fields in Pydantic (V2)
--------------------------------
What is @computed_field?
- In standard Pydantic, regular Python @property getters are NOT included when you
  serialize a model using `.model_dump()` or `.model_dump_json()`.
- `@computed_field` allows you to define properties that are dynamically calculated
  from existing attributes AND automatically included in:
    1. Attribute access (e.g., patient.bmi)
    2. Serialized dictionary outputs (patient.model_dump())
    3. JSON serialization (patient.model_dump_json())
    4. OpenAPI / JSON schema generation in FastAPI responses
"""

from pydantic import BaseModel, Field, computed_field


class Patient(BaseModel):
    name: str = Field(..., description="Name of the patient")
    age: int = Field(..., gt=0, lt=150, description="Age in years")
    weight: float = Field(..., gt=0, description="Weight in kilograms (kg)")
    height: float = Field(..., gt=0, description="Height in meters (m)")

    # -------------------------------------------------------------
    # 1. Computed Field: Body Mass Index (BMI)
    # Formula: weight (kg) / [height (m)]^2
    # The return type hint `-> float` is used by Pydantic for schemas.
    # -------------------------------------------------------------
    @computed_field
    @property
    def bmi(self) -> float:
        """Calculates BMI rounded to 2 decimal places."""
        return round(self.weight / (self.height**2), 2)

    # -------------------------------------------------------------
    # 2. Computed Field referencing another computed field:
    # Notice we can access `self.bmi` directly inside another computed field!
    # -------------------------------------------------------------
    @computed_field
    @property
    def bmi_category(self) -> str:
        """Categorizes patient based on their computed BMI value."""
        calculated_bmi = self.bmi
        if calculated_bmi < 18.5:
            return "Underweight"
        elif calculated_bmi < 24.9:
            return "Normal weight"
        elif calculated_bmi < 29.9:
            return "Overweight"
        else:
            return "Obese"

    # -------------------------------------------------------------
    # 3. Another Example: Birth Year calculation
    # -------------------------------------------------------------
    @computed_field
    @property
    def approx_birth_year(self) -> int:
        """Estimates birth year assuming current year 2026."""
        current_year = 2026
        return current_year - self.age


# ==========================================
# Demonstration & Usage
# ==========================================
if __name__ == "__main__":
    # Instantiate the patient (we only provide name, age, weight, and height)
    patient = Patient(
        name="Nitish",
        age=30,
        weight=75.0,  # 75 kg
        height=1.75,  # 1.75 meters
    )

    print("--- 1. Direct Attribute Access ---")
    print(f"Patient Name:       {patient.name}")
    print(f"Weight / Height:    {patient.weight}kg / {patient.height}m")
    print(f"Computed BMI:       {patient.bmi}")
    print(f"Computed Category:  {patient.bmi_category}")
    print(f"Approx Birth Year:  {patient.approx_birth_year}")
    print()

    print("--- 2. Serialized Dictionary (model_dump()) ---")
    # Notice: 'bmi', 'bmi_category', and 'approx_birth_year' are automatically included!
    patient_dict = patient.model_dump()
    print(patient_dict)
    print()

    print("--- 3. JSON Output (model_dump_json()) ---")
    # Perfect for FastAPI endpoint responses
    print(patient.model_dump_json(indent=2))
