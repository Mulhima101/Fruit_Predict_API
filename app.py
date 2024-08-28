from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    print('hello world')
    return{"hello_world": 'Mulhima'}