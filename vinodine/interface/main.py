from VinoDine.ml_logic import data
from VinoDine.ml_logic import model

#
file_path = "~/code/ArjanAngenent/VinoDine/raw_data/Cleaned_Full_100K_wines.csv"
data_save_path = "~/code/ArjanAngenent/VinoDine/data.pkl"
preprocessor_save_path = "~/code/ArjanAngenent/VinoDine/preprocessor.pkl"
model_save_path = "~/code/ArjanAngenent/VinoDine/model.pkl"

#
wine_df = data.create_binary_df(file_path)
print(wine_df.head())
# X_train, X_test, y_train, y_test = data.create_X_train_y_train(wine_df)
# data.save_data(X_train, X_test, y_train, y_test, data_save_path)

# # Model training and saving:
# model = model.train_model(X_train, y_train)
# model.save_model(model, model_save_path)

# # X_pred
# X_pred=data.create_X_pred()

# # y_pred
# y_pred = model.predict(X_pred)
