import json
from typing import Annotated, Literal, Optional
from fastapi import FastAPI, HTTPException, Path, status
from pydantic import BaseModel, Field

app = FastAPI()


# -------------------------------------------------------------
# Pydantic Model for Update:
# All fields are Optional with default=None so the client can
# update only the specific fields they want to change.
# -------------------------------------------------------------
class PatientUpdate(BaseModel):
    name: Annotated[Optional[str], Field(default=None, description="Name of the patient")]
    city: Annotated[Optional[str], Field(default=None, description="City where the patient is living")]
    age: Annotated[Optional[int], Field(default=None, gt=0, lt=120, description="Age of the patient")]
    gender: Annotated[
        Optional[Literal["male", "female", "others"]],
        Field(default=None, description="Gender of the patient"),
    ]
    height: Annotated[Optional[float], Field(default=None, gt=0, description="Height in meters")]
    weight: Annotated[Optional[float], Field(default=None, gt=0, description="Weight in kgs")]


def load_data():
    with open("patients.json", "r") as f:
        data = json.load(f)
    return data


def save_data(data):
    with open("patients.json", "w") as f:
        json.dump(data, f, indent=2)


# -------------------------------------------------------------
# GET: View all patients
# -------------------------------------------------------------
@app.get("/patients", status_code=status.HTTP_200_OK)
def get_all_patients():
    return load_data()


# -------------------------------------------------------------
# GET: View single patient by ID
# -------------------------------------------------------------
@app.get("/patient/{patient_id}", status_code=status.HTTP_200_OK)
def get_patient(
    patient_id: Annotated[str, Path(description="ID of the patient to view", examples=["P001"])]
):
    data = load_data()
    if patient_id not in data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient with ID '{patient_id}' not found.",
        )
    return data[patient_id]


# -------------------------------------------------------------
# PUT: Update an existing patient's data
# - Uses status code 200 OK
# - Uses 404 NOT FOUND if patient does not exist
# - Uses exclude_unset=True to only update provided fields
# -------------------------------------------------------------
@app.put("/patient/{patient_id}", status_code=status.HTTP_200_OK)
def update_patient(
    patient_id: Annotated[str, Path(description="ID of the patient to update", examples=["P001"])],
    patient_update: PatientUpdate,
):
    data = load_data()

    # 1. Check if the patient exists
    if patient_id not in data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient with ID '{patient_id}' not found.",
        )

    # 2. Extract only fields that were actually passed by user (ignoring unset/None fields)
    update_data = patient_update.model_dump(exclude_unset=True)

    if not update_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No fields provided for update.",
        )

    # 3. Update the existing patient record
    data[patient_id].update(update_data)

    # 4. Save updated dictionary back to patients.json
    save_data(data)

    return {
        "message": "Patient updated successfully",
        "patient": data[patient_id],
    }


# -------------------------------------------------------------
# DELETE: Remove an existing 
# - Uses status code 200 OK
# - Uses 404 NOT FOUND if patient does not exist
# -------------------------------------------------------------
@app.delete("/patient/{patient_id}", status_code=status.HTTP_200_OK)
def delete_patient(
    patient_id: Annotated[str, Path(description="ID of the patient to delete", examples=["P001"])]
):
    data = load_data()

    # 1. Check if patient exists
    if patient_id not in data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient with ID '{patient_id}' not found.",
        )

    # 2. Remove the patient from the dictionary
    deleted_patient = data.pop(patient_id)

    # 3. Save modified data back to patients.json
    save_data(data)

    return {
        "message": "Patient deleted successfully",
        "deleted_patient": deleted_patient,
    }
