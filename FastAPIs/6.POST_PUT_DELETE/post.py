import json
from typing import Annotated, Literal
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

app = FastAPI()


class Patient(BaseModel):
    id: Annotated[str, Field(..., description="ID of the patient", examples=["P001"])]
    name: Annotated[str, Field(..., description="Name of the patient")]
    city: Annotated[str, Field(..., description="City where the patient is living")]
    age: Annotated[int, Field(..., gt=0, lt=120, description="Age of the patient")]
    gender: Annotated[
        Literal["male", "female", "others"],
        Field(..., description="Gender of the patient"),
    ]
    height: Annotated[
        float, Field(..., gt=0, description="Height of the patient in mtrs")
    ]
    weight: Annotated[
        float, Field(..., gt=0, description="Weight of the patient in kgs")
    ]


def load_data():
    with open("patients.json", "r") as f:
        data = json.load(f)
    return data


def save_data(data):
    with open("patients.json", "w") as f:
        json.dump(data, f, indent=2)


# 1. status_code=status.HTTP_201_CREATED indicates a resource was successfully created
@app.post("/create", status_code=status.HTTP_201_CREATED)
def create_patient(patient: Patient):
    data = load_data()

    # 2. HTTPException raises error if patient ID already exists
    if isinstance(data, dict):
        if patient.id in data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Patient with ID '{patient.id}' already exists.",
            )
        data[patient.id] = patient.model_dump()
    elif isinstance(data, list):
        for item in data:
            if item.get("id") == patient.id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Patient with ID '{patient.id}' already exists.",
                )
        data.append(patient.model_dump())

    save_data(data)
    return {"message": "Patient created successfully", "patient": patient}