from vinodine.ml_logic import data
import pandas as pd
from vinodine.ml_logic.model import create_X_train_y_train, train_model, save_model, open_model, pred

#
file_path = "~/code/ArjanAngenent/VinoDine/raw_data/Cleaned_Full_100K_wines.csv"

model_save_path = "model.pkl"

wine_df = data.create_binary_df(file_path)
X_train, X_test, y_train, y_test = create_X_train_y_train(wine_df)

# Model training and saving:
model = train_model(X_train, y_train)
save_model(model, model_save_path)

if __name__ == '__main__':
    model = open_model(model_save_path)



    # X_pred
    X_pred = pd.DataFrame.from_dict({'Type': ["Red"],
                                        'Body': ["Full-bodied"],
                                        'Acidity': ['High'],
                                        'ABV': [7.8]},
                                    orient='columns')

    y_pred = pred(model, X_pred)
    # y_pred
    print(y_pred)
