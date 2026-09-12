from pydantic import BaseModel

#Address model
class Address(BaseModel):
    city:str
    state:str
    pincode:str

#Patient model
class Patient(BaseModel):
    name:str
    gender:str
    age:int
    address:Address

address_dict={'city':'hyderabad','state':'telangana','pincode':'500001'}
address1=Address(**address_dict)
print(address1)

patient_dict={'name':'nitish','gender':'male','age':21,'address':address_dict}
patient1=Patient(**patient_dict)
print(patient1)
print(patient1.address.pincode)
print(patient1.address.city)

# Better organization of related data (e.g., vitals, address, insurance)

# Reusability: Use Vitals in multiple models (e.g., Patient, MedicalRecord)

# Readability: Easier for developers and API consumers to understand

# Validation: Nested models are validated automatically-no extra work needed
