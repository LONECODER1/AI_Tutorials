from pydantic import AnyUrl
from pydantic import BaseModel,EmailStr
from typing import List,Dict,Optional
class Patient(BaseModel):
    name:str
    email:EmailStr
    age:int
    disease:Optional[List[str]] 
    disease:str
    linkedin_url:AnyUrl
    
