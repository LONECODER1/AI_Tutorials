from pydantic import BaseModel
from typing import List,Dict,Optional

class Patient(BaseModel):
    name:str
    age:int
    email:str
    disease:Optional[List[str]]=None
    married:bool=False
    contact_details:Dict[str,str]

def insert_patient_data(patient:Patient):
    print(patient.name)    
    print(patient.age)
    print(patient.email)
    print(patient.disease)
    print(patient.married)

    return {"message":"Patient data inserted successfully"}    

if __name__ == "__main__":
    sample_patient = Patient(
        name="Alice",
        age=28,
        email="alice@example.com",
        disease=["Flu"],
        married=False,
        contact_details={"phone": "123-456-7890"}
    )
    
    insert_patient_data(sample_patient)
