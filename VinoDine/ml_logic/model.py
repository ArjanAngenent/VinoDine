import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.multiclass import OneVsRestClassifier
import pickle


def train_model(X_train, y_train):
    # Create binary classifier
    # Define which columns need to be encoded
    cat_cols = make_column_selector(dtype_include='object')
    num_cols = make_column_selector(dtype_include='number')
    cat_pre = make_pipeline(OneHotEncoder(sparse_output=False, handle_unknown='ignore'),
                            MinMaxScaler())
    cat_num = MinMaxScaler()

    # Create preprocessor pipeline
    preprocessing = make_column_transformer((cat_pre, cat_cols),(cat_num, num_cols))

    binary_classifier = SGDClassifier(max_iter=500, random_state=42)

    ova_classifier = OneVsRestClassifier(binary_classifier)

    pipeline = make_pipeline(preprocessing, ova_classifier)
    return pipeline.fit(X_train, y_train)

def create_X_train_y_train(df, test_size=0.3):
    X = df[['Type','Body','Acidity', 'ABV']]
    y = df.drop(columns=['Type','Elaborate','Body','Acidity', 'ABV'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=test_size)
    return X_train, X_test, y_train, y_test

def save_model(model, model_save_path):
    # Save the model as a pickle file
    with open(model_save_path, 'wb') as f:
        pickle.dump(model, f)

def open_model(model_open_path):
    # Open the model as a pickle file
    with open(model_open_path, "rb") as f:
        model = pickle.load(f)
    return model
