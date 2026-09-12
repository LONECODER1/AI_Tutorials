from pydantic import BaseModel

class Address(BaseModel):
    city:str
    state:str
    pin:int

class Patient(BaseModel):
    name:str
    age:int
    address:Address

patient1=Patient(
    name='nitish',
    age=21,
    address={
        'city':'hyderabad',
        'state':'telangana',
        'pin':500001
    }
)

dict_output=patient1.model_dump()
print(dict_output)

json_output=patient1.model_dump_json(indent=2)
print(json_output)

# ------------------------------
# toExclude
# ------------------------------
dict_output_no_name=patient1.model_dump(exclude={'name'})
print(dict_output_no_name)

json_output_no_name=patient1.model_dump_json(exclude={'name'},indent=2)
print(json_output_no_name)