from google.protobuf.internal.encoder import StringEncoder
from pydantic import BaseModel
class Patient(BaseModel):
    name:str
    gender:str
    age:int