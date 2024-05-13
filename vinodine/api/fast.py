import pandas as pd
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from vinodine.ml_logic.model import open_model

app = FastAPI()

model_open_path="model.pkl"
app.state.model = open_model(model_open_path)

@app.get("/predict")
def predict(
        Type: str,
        ABV: float,
        Body: str,
        Acidity: str
    ):

    # Make a single food prediction.
    X_pred = pd.DataFrame.from_dict({'Type': [Type],
                                    'Body': [Body],
                                    'Acidity': [Acidity],
                                    'ABV': [ABV]},
                                   orient='columns')

    y_pred = app.state.model.predict(X_pred)

    foods = ['Beef', 'CuredMeat', 'GameMeat', 'Lamb', 'Pasta', 'Pork', 'Poultry', 'RichFish', 'Shellfish', 'Veal', 'Vegetarian']
    foods_index = np.where(y_pred[0]==1)[0].tolist()
    foods_to_choose = []
    for i in foods_index:
        foods_to_choose.append(foods[i])

    return {"foods": foods_to_choose}

@app.get("/")
def root():
    # $CHA_BEGIN
    return dict(greeting='Hello')
    # $CHA_END
