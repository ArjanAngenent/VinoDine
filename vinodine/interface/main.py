from vinodine.ml_logic.data import create_binary_df
from vinodine.ml_logic.model import create_X_train_y_train, train_model, save_model, open_model, create_X_pred, pred
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

file_path = "~/code/ArjanAngenent/VinoDine/raw_data/Cleaned_Full_100K_wines.csv"

model_save_path = 'model.pkl'

# Loading and binary encoding source data frame("grape_column": retrieving last column for binary encoded grapes to create X)
wine_df, grape_colum = create_binary_df(file_path)

X_train, X_test, y_train, y_test = create_X_train_y_train(wine_df, grape_column=grape_colum)

# Model training and saving
model = train_model(X_train, y_train)
save_model(model, model_save_path)

if __name__ == '__main__': # remove "#" in python file and indent following lines
    model = open_model(model_save_path)

    y_pred = model.predict(X_test)

    # Print classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Evaluate performance
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # create an new X_pred with manual input
    X_pred = create_X_pred(Type='Red',
                        Body='Full-bodied',
                        Acidity='High',
                        ABV=14.5,
                        grapes=['Cabernet Sauvignon'])

    # predict foods for new input
    y_pred_foods = pred(model, X_pred)

    print(y_pred_foods)
