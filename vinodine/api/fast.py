from vinodine.ml_logic.model import open_model, create_X_pred, pred
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List

app = FastAPI()
app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_methods=["*"],
                   allow_headers=["*"])


model_open_path = 'model.pkl'
app.state.model = open_model(model_open_path)


@app.get('/')
def root():
    return {'greeting': 'Hello'}


@app.get('/predict')
def predict(Type: str,
            ABV: float,
            Body: str,
            Acidity: str,
            grapes: List[str] = Query('grape')):
    # create managebale X_pred for model from API request
    X_pred = create_X_pred(Type = Type,
                           ABV = ABV,
                           Body = Body,
                           Acidity = Acidity,
                           grapes = grapes)

    # return suggested foods based on pretrained model and X_pred created from API request
    y_pred_foods = pred(app.state.model, X_pred)

    return y_pred_foods
