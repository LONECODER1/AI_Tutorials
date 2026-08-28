from fastapi import FastAPI,Path
# import the FastAPI lib
import json

app = FastAPI()
# create an object instace for fastAPI

def load_data():
    with open('patients.json','r') as f:
        data=json.load(f)
    return data  

@app.get("/")
def hello():
    return {'message':'PMS API'}

@app.get("/about")
def about():
    return {'message':'This is an API to manage a PMS'}

@app.get('/view')
def view():
    data = load_data()
    return data
@app.get('/patient/{patient_id}')
def view_patient(patient_id:str=Path(...,description='Id of the patient',example='P001')):
    #load all the patients
    data=load_data()
    if patient_id in data:
        return data[patient_id]
    return {'error':'patient not found'}
    