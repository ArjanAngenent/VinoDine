import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


def create_binary_df(file_path):

    '''
    Load the original dataframe,
    binary encode the 'Grapes' (part of features) and 'Harmonize' (becomes multi-label target y),
    remove all columns not needed for model training,
    return the binary encoded and cleaned dataframe
    '''

    #binary encoder for 'Harmonize' column (multi-label target)
    mlb_harm = MultiLabelBinarizer(sparse_output=False)
    #binary encoder for 'Grape' column (feature)
    mlb_grape = MultiLabelBinarizer(sparse_output=False)

    #read in origina data from file path
    wine_df = pd.read_csv(file_path)

    #Drop addional columns not used for model
    wine_df = wine_df.drop(columns=['WineName', 'WineID','Code','Country','RegionID','RegionName','WineryID','Website','Vintages', 'WineryName'])

    # Binary encode grapes
    wine_df_bin = wine_df.join(pd.DataFrame(
        mlb_grape.fit_transform(eval(element) for element in wine_df.Grapes),
        index=wine_df.index,
        columns=mlb_grape.classes_
        ))
    wine_df_bin.drop(columns=['Grapes'], inplace=True)

    # Create a list of the kind of grapes that are mentioned less then 2.000 times
    grapes_list = wine_df_bin.iloc[:,16:].sum() # sum the number of times a grape is mentioned via column
    final_column_grapes = wine_df_bin.shape[1]

    # Binary encode Harmonize(kinds of food)
    wine_df_bin = wine_df_bin.join(pd.DataFrame(
        mlb_harm.fit_transform(eval(element) for element in wine_df.Harmonize),
        index=wine_df.index,
        columns=mlb_harm.classes_
        ))
    wine_df_bin.drop(columns=['Harmonize'], inplace=True)

    # Create a list of the kind of grapes that are mentioned less then 2.000 times
    harm_list = wine_df_bin.iloc[:,(final_column_grapes+1):].sum() # sum the number of times a food is mentioned via column
    harm_to_drop = harm_list[harm_list<=15_000].index.to_list() # create a list withe kind of food mentioned less then 50 times
    wine_df_bin.drop(columns=harm_to_drop, inplace=True) # drop columns with food not mentioned more then 50 times
    wine_df_bin = wine_df_bin[wine_df_bin.iloc[:,(final_column_grapes+1):].eq(1).any(axis=1)] # drop wines which are not represented by a food anymore

    return wine_df_bin, final_column_grapes
