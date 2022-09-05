from turtle import Vec2D
from fastapi import FastAPI
import pickle
from pydantic import BaseModel
import pandas as pd
from typing import List
import numpy as np

model = pickle.load(open(r'app/Modelo_Credit.sav', 'rb'))

app = FastAPI()


class Credit(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

class ItemIn(BaseModel):
    variaveis : List[Credit]


@app.post('/')
def predict(thisdict: ItemIn):
    thisdict2 = thisdict.dict()
    guardar = pd.DataFrame(thisdict2['variaveis'])
    modelo = model.predict(guardar).tolist()
    return {'prediction': modelo} 
