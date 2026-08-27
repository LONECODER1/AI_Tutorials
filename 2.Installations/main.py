from fastapi import FastAPI
# import the FastAPI lib

app = FastAPI()
# create an object instace for fastAPI

@app.get("/")
def hello():
    return {'message':'Hello World'}

@app.get("/about")
def about():
    return {'message':'This is my firts fastAPI application'}