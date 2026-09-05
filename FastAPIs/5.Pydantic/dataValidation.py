from pydantic import Field

# used for setting limit and metadata
from pydantic import BaseModel, EmailStr, AnyUrl
from typing import List, Dict, Optional, Annotated


class Patient(BaseModel):
    name: Annotated[
        str,
        Field(
            ...,
            min_length=2,
            max_length=15,
            title="Name of the patient",
            description="Give the name of the patient in less than 50 chars",
            examples=["Nitish", "Amit"],
        ),
    ]
    email: Annotated[EmailStr, Field(title="email")]
    age: Annotated[int, Field(..., gt=1, lt=150, title="age")]
    disease: Optional[List[str]] = Field(max_length=5)
    weight: float = Field(..., gt=40, lt=150)
    disease: str
    linkedin_url: AnyUrl
