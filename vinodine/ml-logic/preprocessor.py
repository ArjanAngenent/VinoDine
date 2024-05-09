import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

y = wine_df['Harmonize_Combined']
X = wine_df[['Type', 'ABV', 'Body', 'Acidity']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# features to select: Type, Elaborate, Grapes, ABV, Body, Acidity
# target: harmonize
#y = wine_df['Harmonize']
#X = wine_df[['Type', 'ABV', 'Body', 'Acidity']]
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Create OneHotEncoder





from colorama import Fore, Style
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
import pickle
def preprocess_features(X: pd.DataFrame) -> np.ndarray:
    def create_sklearn_preprocessor() -> ColumnTransformer:
        """
        Scikit-learn pipeline that transforms a cleaned dataset of shape (_, 7)
        into a preprocessed one of fixed shape (_, 65).
        Stateless operation: "fit_transform()" equals "transform()".
        """
        ohe = OneHotEncoder(sparse_output=False)
        # Create Ordinal Enoder for Type with 6 categories ['Red', 'White', 'Rose', 'Dessert/Port', 'Dessert', 'Sparkling']
        # Create Ordinal Enoder for Body with 5 categories ['Very full-bodied', 'Full-bodied', 'Medium-bodied', 'Light-bodied', 'Very light-bodied']
        # Create Ordinal Enoder for Acidity with 3 categories ['High', 'Medium', 'Low']
        Type_categories = ['Red', 'White', 'Rose', 'Dessert/Port', 'Dessert', 'Sparkling']
        Body_categories = ['Very full-bodied', 'Full-bodied', 'Medium-bodied', 'Light-bodied', 'Very light-bodied']
        Acidity_categories = ['High', 'Medium', 'Low']

        # ordinal_encoder = OrdinalEncoder(categories=[['High', 'Medium', 'Low']])
        # Create StandardScaler for ABV(numerical feeature)
        min_max_scaler = MinMaxScaler()

        # fit_transform encoders and scalers to X_train
        #X_train[ohe.get_feature_names_out()] = ohe.fit_transform(X_train[['Type','Body','Acidity']])
        # X_train['encoded_acidicity'] = ordinal_encoder.fit_transform(X_train[['Acidity']])
        #X_train['ABV'] = min_max_scaler.fit_transform(X_train[['ABV']])
        #X_train.drop(columns=['Type','Body', 'Acidity'], inplace=True)
        # transform X_test with encoders and scalers
        #X_test[ohe.get_feature_names_out()] = ohe.transform(X_test[['Type','Body','Acidity']])
        # X_test['encoded_acidicity'] = ordinal_encoder.transform(X_test[['Acidity']])
        #X_test['ABV'] = min_max_scaler.transform(X_test[['ABV']])
        #X_test.drop(columns=['Type', 'Body', 'Acidity'], inplace=True)

        ABV_pipe = make_pipeline(
            MinMaxScaler(
                categories=Type_categories,
                handle_unknown="ignore",
                sparse_output=False
            )
        )
        Type_pipe = make_pipeline(
            OneHotEncoder(
                categories=Type_categories,
                handle_unknown="ignore",
                sparse_output=False
            )
        )
        Body_pipe = make_pipeline(
            OneHotEncoder(
                categories=Body_categories,
                handle_unknown="ignore",
                sparse_output=False
            )
        )
        Acidity_pipe = make_pipeline(
            OneHotEncoder(
                categories=Acidity_categories,
                handle_unknown="ignore",
                sparse_output=False
            )
        )
        # COMBINED PREPROCESSOR
        final_preprocessor = ColumnTransformer(
            [
                ("ABV_scaler", ABV_pipe, ["ABV"]),
                ("Type_preproc", Type_pipe, ["Type"]),
                ("Body_preproc", Body_pipe, ["Body"]),
                ("Acidity_preproc", Acidity_pipe, ["Acidity"]),
            ],
            n_jobs=-1,
        )
        return final_preprocessor
    print(Fore.BLUE + "\nPreprocessing features..." + Style.RESET_ALL)
    preprocessor = create_sklearn_preprocessor()
    X_processed = preprocessor.fit_transform(X)
    file_path = '~/code/ArjanAngenent/VinoDine/X_processed.pkl'
    # Save the preprocessed data as a pickle file
    with open(file_path, 'wb') as f:
        pickle.dump(preprocessor, f)
    print(":white_check_mark: X_processed, with shape", X_processed.shape)
    return X_processed
