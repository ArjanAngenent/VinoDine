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

def create_binary_df(file_path):
    mlb_harm = MultiLabelBinarizer(sparse_output=False)
    mlb_grape = MultiLabelBinarizer(sparse_output=False)

    wine_df = pd.read_csv(file_path)

    wine_df.drop(columns=['Grapes'], inplace=True)

    # Binary encode Harmonize(kinds of food)
    wine_df_bin = wine_df.join(pd.DataFrame(
        mlb_harm.fit_transform(eval(element) for element in wine_df.Harmonize),
        index=wine_df.index,
        columns=mlb_harm.classes_
        ))
    wine_df_bin.drop(columns=['Harmonize'], inplace=True)
    
    # Create a list of the kind of grapes that are mentioned less then 2.000 times
    harm_list = wine_df_bin.iloc[:,15:].sum() # sum the number of times a food is mentioned via column
    harm_to_drop = harm_list[harm_list<=15_000].index.to_list() # create a list withe kind of food mentioned less then 50 times
    wine_df_bin.drop(columns=harm_to_drop, inplace=True) # drop columns with food not mentioned more then 50 times
    wine_df_bin = wine_df_bin[wine_df_bin.iloc[:,15:].eq(1).any(axis=1)] # drop wines which are not represented by a food anymore

    #Drop addional columns not used for model
    wine_df_bin_cleaned = wine_df_bin.drop(columns=['WineName', 'WineID','Code','Country','RegionID','RegionName','WineryID','Website','Vintages', 'WineryName'])

    return wine_df_bin_cleaned


