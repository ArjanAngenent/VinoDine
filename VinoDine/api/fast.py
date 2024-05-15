import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.multiclass import OneVsRestClassifier
from imblearn.over_sampling import RandomOverSampler
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pickle
from vinodine.ml_logic.model import open_model
from vinodine.ml_logic.data import create_X_pred

app = FastAPI()

model_open_path="~/code/VinoDine/"
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
    return dict(greeting=“Hello”)
    # $CHA_END





# 'Varietal/100%',
y_pred = model.predict(create_X_pred('Red', 'Full-bodied', 'Medium', 7.8))

# 'Varietal/100%',
show_foods(y_train, y_pred)
