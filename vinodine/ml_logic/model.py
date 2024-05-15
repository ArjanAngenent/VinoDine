from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

from skmultilearn.problem_transform import LabelPowerset

import pickle

def create_X_train_y_train(df, grape_column, test_size=0.3):

    '''
    Read in transformed data frame and the final column for grapes (last feature column of data frame)
    Return X_train, X_test, y_train, y_test
    '''

    X = df.iloc[:,:grape_column-1]
    X = X.drop(columns = 'Elaborate')
    y = df.iloc[:,grape_column:]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=test_size)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):

    '''
    Create pipeline for feature transformation and
    model training
    '''

    # Create binary classifier
    # Define which columns need to be encoded
    cat_cols = make_column_selector(dtype_include='object')
    num_cols = make_column_selector(dtype_include='number')
    cat_pre = make_pipeline(OneHotEncoder(sparse_output=False, handle_unknown='ignore'),
                            MinMaxScaler())
    cat_num = MinMaxScaler()

    # Create preprocessor pipeline
    preprocessing = make_column_transformer((cat_pre, cat_cols),(cat_num, num_cols))

    classifier = LabelPowerset(RandomForestClassifier(max_features=1, min_samples_split=10,
                                                      n_jobs=-1, random_state=42))

    pipeline = make_pipeline(preprocessing, classifier)
    return pipeline.fit(X_train, y_train)

def save_model(model, model_save_path):
    # Save the model as a pickle file
    with open(model_save_path, 'wb') as f:
        pickle.dump(model, f)

def open_model(model_open_path):
    # Open the model as a pickle file
    with open(model_open_path, "rb") as f:
        model = pickle.load(f)
    return model

def create_X_pred(Type: str,
            ABV: float,
            Body: str,
            Acidity: str,
            grapes: list):

    '''
    Return a X_pred readable by the model by populating all grape columns not
    listed with 0 and the ones listed with 0
    '''

    X_pred = pd.DataFrame.from_dict({'Type': [Type],
                                        'Body': [Body],
                                        'Acidity': [Acidity],
                                        'ABV': [ABV]},
                                    orient='columns')

    # list of all grape columns (binary encoded)
    grapes_columns = ['"BiancodAlessano"', '"LAcadieBlanc"', '"LendelEl"', '"LoindelOeil"', '"NerodAvola"', '"PineauDAunis"', '"RoussetteDAyze"', '"TrebbianodAbruzzo"', 'Abbuoto', 'Abouriou', 'Abrostine', 'Acolon', 'Agiorgitiko', 'Aglianico', 'Aidani', 'Airen', 'Albalonga', 'Albana', 'Albanella', 'Albariño', 'Albarola', 'Albarossa', 'AlbarínBlanco', 'Albillo', 'AlbilloCrimean', 'AlbilloMayor', 'AlbillodeAlbacete', 'Aleatico', 'AlfrocheiroPreto', 'Alibernet', 'AlicanteBouschet', 'AlicanteGanzin', 'Aligoté', 'Altesse', 'Alvarelhão', 'Alvarinho', 'Amigne', 'Ancellotta', 'Ansonica', 'AntãoVaz', 'Aragonez', 'Aramon', 'Arbane', 'Areni', 'Argaman', 'Arinarnoa', 'Arinto', 'ArintodeBucelas', 'ArintodosAçores', 'Arneis', 'Arnsburger', 'Arriloba', 'AspiranBouschet', 'AsprinioBianco', 'AssarioBranco', 'Assyrtiko', 'Athiri', 'Aurore', 'Avanà', 'Avesso', 'Avgoustiatis', 'AzalBranco', 'AzalTinto', 'Babić', 'Bacchus', 'BacoNoir', 'Baga', 'Barbarossa', 'Barbera', 'Barcelo', 'Barsaglina', 'BastardoMagarachsky', 'Batoca', 'Bellone', 'Bianca', 'Biancame', 'BianchettaTrevigiana', 'Biancolella', 'Bical', 'BlackQueen', 'Blauburger', 'Blauburgunder', 'BlauerPortugieser', 'Blaufränkisch', 'BoalBranco', 'Bobal', 'Bogazkere', 'BombinoBianco', 'BombinoNero', 'Bonamico', 'Bonarda', 'Bordô', 'Borraçal', 'Bosco', 'Bourboulenc', 'Bovale', 'Brachetto', 'Braquet', 'Braucol', 'Brianna', 'Bronner', 'BrunArgenté', 'Bruñal', 'Bual', 'BudaiZöld', 'Bukettraube', 'BurgundMare', 'BusuioacadeBohotin', 'BăbeascăNeagră', 'CabernetBlanc', 'CabernetCortis', 'CabernetCubin', 'CabernetDorsa', 'CabernetFranc', 'CabernetJura', 'CabernetMitos', 'CabernetRuby', 'CabernetSauvignon', 'CabernetSeverny', 'Cagnulari', 'CaiñoBlanco', 'CaiñoTinto', 'CalabresediMontenuovo', 'Caladoc', 'Calkarasi', 'Callet', 'Camarate', 'CanaioloBlanco', 'CanaioloNero', 'Cannonau', 'Carignan/Cariñena', 'Carmenère', 'Carricante', 'Casavecchia', 'Cascade', 'Casetta', 'Castelão', 'CatarrattoBianco', 'Catawba', 'CayugaWhite', 'Cencibel', 'Centesiminio', 'CercealBranco', 'Cesanese', 'Chambourcin', 'Chancellor', 'Charbono', 'Chardonel', 'Chardonnay', 'ChardonnayMusqué', 'Chasan', 'Chasselas', 'Chatus', 'Chenanson', 'CheninBlanc', 'Chinuri', 'Cienna', 'Ciliegiolo', 'Cinsault', 'Clairette', 'Cococciola', 'CodadiVolpeBianca', 'Colobel', 'Colombard', 'Coloraillo', 'ColorinodelValdarno', 'Concord', 'CorintoNero', 'Cornalin', 'Cornifesto', 'CorotNoir', 'Cortese', 'Corvina', 'Corvinone', 'Couderc', 'Counoise', 'CriollaGrande', 'Croatina', 'Crouchen', 'Cynthiana', 'CôdegadeLarinho', 'Côt', 'Dafni', 'Dakapo', 'DeChaunac', 'Debina', 'Diagalves', 'Dimiat', 'Dimrit', 'Dindarella', 'Diolinoir', 'Dolcetto', 'Domina', 'DonaBlanca', 'DonzelinhoBranco', 'DonzelinhoTinto', 'Dornfelder', 'Drupeggio', 'Dunkelfelder', 'Duras', 'Durella', 'Durif', 'DzvelshaviObchuri', 'Edelweiss', 'Egiodola', 'Ehrenfelser', 'EmeraldRiesling', 'Emir', 'Enantio', 'Encruzado', 'Erbaluce', 'Espadeiro', 'Falanghina', 'FalanghinaBeneventana', 'Famoso', 'Favorita', 'Fenile', 'FerServadou', 'FernãoPires', 'FeteascaAlba', 'FeteascaNeagra', 'FeteascaRegala', 'Fiano', 'Flora', 'FogliaTonda', 'Fokiano', 'Folgasao', 'FolleBlanche', 'FonteCal', 'Fragolino', 'Francusa', 'Frappato', 'Fredonia', 'Freisa', 'Friulano/Sauvignonasse', 'Frontenac', 'FruhroterVeltliner', 'Frühburgunder', 'Fumin', 'FuméBlanc', 'Furmint', 'Gaglioppo', 'Gaidouria', 'Galotta', 'Gamaret', 'GamayNoir', 'GamayTeinturierdeBouze', 'GambadiPernice', 'Garanoir', 'Garganega', 'Garnacha', 'GarnachaBlanca', 'GarnachaPeluda', 'GarnachaRoja', 'GarnachaTinta', 'GarnachaTintorera', 'GarridoFino', 'GelberMuskateller', 'Gewürztraminer', 'Gigiac', 'Ginestra', 'Girgentina', 'GiròBlanc', 'Glera/Prosecco', 'Godello', 'GoldTraminer', 'Goldburger', 'Golubok', 'Gorgollasa', 'GoruliMtsvane', 'Gouveio', 'GouveioReal', 'Graciano', 'GrandNoir', 'GrasadeCotnari', 'Grauburgunder', 'Grecanico', 'Grechetto', 'GrechettoRosso', 'Greco', 'GrecoBianco', 'GrecoNero', 'Grenache', 'GrenacheBlanc', 'GrenacheGris', 'Grignolino', 'Grillo', 'Gringet', 'Grolleau', 'Groppello', 'GrosManseng', 'GrosVerdot', 'GrünerVeltliner', 'Guardavalle', 'Gutedel', 'Hanepoot', 'Helios', 'Hibernal', 'HondarrabiBeltza', 'HondarrabiZuri', 'HumagneBlanche', 'HumagneRouge', 'Huxelrebe', 'Hárslevelű', 'IncrocioManzoni', 'Inzolia', 'IrsaiOliver', 'Isabella', 'Jacquère', 'Jaen', 'Jampal', 'Johannisberg', 'Johanniter', 'JuanGarcia', 'Kabar', 'Kadarka', 'Kakhet', 'Kakotrygis', 'KalecikKarasi', 'Kangun', 'Karasakiz', 'Karmahyut', 'Katsano', 'Keratsuda', 'Kerner', 'Khikhvi', 'Királyleányka', 'Kisi', 'Klevner', 'KokurBely', 'Koshu', 'Kotsifali', 'KrasnostopAnapsky', 'KrasnostopZolotovsky', 'Kratosija', 'Krstac', 'Kydonitsa', 'Kékfrankos', 'Lacrima', 'Lafnetscha', 'Lagrein', 'Lambrusco', 'Lampia', 'LandotNoir', 'Lauzet', 'Leanyka', 'Lefkada', 'Lemberger', 'Lenoir', 'LeonMillot', 'Liatiko', 'Limnio', 'Limniona', 'ListanNegro', 'Lorena', 'Loureiro', 'Macabeo', 'MadeleineAngevine', 'MaglioccoCanino', 'Malagouzia', 'Malbec', 'MalboGentile', 'Malvar', 'Malvasia', 'MalvasiaBiancaLunga', 'MalvasiaFina', 'MalvasiaIstriana', 'MalvasiaNera', 'MalvasiadelLazio', 'MalvasiadiCandia', 'MalvasiadiLipari', 'MalvasiadiSchierano', 'MalvazijaIstarska', 'Mammolo', 'Mandilaria', 'Mandón', 'Manseng', 'Manteudo', 'MantoNegro', 'ManzoniBianco', 'Maratheftiko', 'MarechalFoch', 'MariaGomes', 'Marmajuelo', 'Marquette', 'Marsanne', 'Marselan', 'Marufo', 'Marzemino', 'Mataro', 'MaturanaBlanca', 'MaturanaTinta', 'MauzacBlanc', 'MauzacNoir', 'Mavro', 'MavroKalavritino', 'Mavrodafni', 'Mavrotragano', 'MavroudiArachovis', 'Mavrud', 'Mayolet', 'Mazuelo', 'Melnik', 'Melody', 'MelondeBourgogne', 'Mencia', 'Menoir', 'Merlot', 'Merseguera', 'Michet', 'Millot-Foch', 'MisketCherven', 'MisketVrachanski', 'ModrýPortugal', 'Molinara', 'Mollard', 'Monastrell', 'MondeuseNoire', 'Monica', 'Montepulciano', 'Montuni', 'Moradella', 'Morava', 'Morellino', 'Morenillo', 'Moreto', 'Morio-Muskat', 'Moristel', 'Moschofilero', 'Moschomavro', 'Mouhtaro', 'Mourisco', 'Mourvedre', 'MtsvaneKakhuri', 'Muscadelle', 'Muscadine', 'Muscardin', 'Muscat/MoscatelGalego', 'Muscat/MoscatelRoxo', 'Muscat/MoscateldeGranoMenudo', 'Muscat/MoscatelloSelvatico', 'Muscat/Moscato', 'Muscat/MoscatoBianco', 'Muscat/MoscatoGiallo', 'Muscat/MoscatoRosa', 'Muscat/MoscatodiScanzo', 'Muscat/Muscatel', 'Muscat/MuskatMoravsky', 'MuscatBaileyA', 'MuscatBlack', 'MuscatBlanc', 'MuscatEarly', 'MuscatGolden', 'MuscatNoir', 'MuscatOrange', 'MuscatOttonel', 'MuscatValvin', 'MuscatYellow', 'MuscatofAlexandria', 'MuscatofFrontignan', 'MuscatofHamburg', 'MuscatofSetúbal', 'MustoasadeMaderat', 'Müller-Thurgau', 'Narince', 'Nascetta', 'Nasco', 'Nebbiolo', 'Negoska', 'NegraraTrentino', 'NegraraVeronese', 'Negrette', 'Negroamaro', 'NegrudeDragasani', 'NerelloCappuccio', 'NerelloMascalese', 'NerettaCuneese', 'NeroBuonodiCori', 'NerodiTroia', 'Neuburger', 'Niagara', 'NiagaraBlanc', 'Nieddera', 'Nielluccio', 'Noble', 'Nocera', 'Noiret', 'Norton', 'Nosiola', 'Nouvelle', 'Nuragus', 'Ojaleshi', 'OlaszRizling', 'Ondenc', 'Orion', 'OrleansGelb', 'Ortega', 'Ortrugo', 'Oseleta', 'OtskhanuriSapere', 'Padeiro', 'Pagadebit', 'Palava', 'PallagrelloBianco', 'PallagrelloNero', 'Palomino', 'Pamid', 'Pampanuto', 'Parellada', 'Parraleta', 'Pascale', 'Passerina', 'Pavana', 'País/Mission', 'Pecorino', 'Pederna', 'Pedral', 'PedroXimenez', 'Pelaverga', 'Peloursin', 'Perera', 'Perle', 'Perricone', 'Perrum', 'PetitCourbu', 'PetitManseng', 'PetitMeslier', 'PetitRouge', 'PetitVerdot', 'PetiteArvine', 'PetiteMilo', 'PetitePearl', 'PetiteSirah', 'Peverella', 'Phoenix', 'Picardan', 'PiccolaNera', 'Picolit', 'PicpoulBlanc', 'Piedirosso', 'Pigato', 'Pignoletto', 'Pignolo', 'Pinenc', 'PinotAuxerrois', 'PinotBlanc', 'PinotGrigio', 'PinotGris', 'PinotMeunier', 'PinotNero', 'PinotNoir', 'Pinotage', 'PiquepoulBlanc', 'PiquepoulNoir', 'PlavacMali', 'PolleraNera', 'PosipBijeli', 'Poulsard', 'Premetta', 'Prensal', 'PretoMartinho', 'PrietoPicudo', 'Primitivo', 'Prié', 'Procanico', 'Prokupac', 'PrugnoloGentile', 'Pugnitello', 'Pulcinculo', 'Rabigato', 'RabodeOvelha', 'RabosoPiave', 'RabosoVeronese', 'Ramisco', 'Rebo', 'Refosco', 'RefoscodalPeduncoloRosso', 'Regent', 'Reichensteiner', 'RibollaGialla', 'Riesel', 'Rieslaner', 'Riesling', 'RieslingItálico', 'RieslingRenano', 'Ripolo', 'Rivaner', 'Rkatsiteli', 'Robola', 'Roditis', 'Roesler', 'Rolle/Rollo', 'Romeiko', 'Romé', 'Rondinella', 'Rondo', 'Roobernet', 'Roscetto', 'Rosetta', 'Rossese', 'Rossignola', 'Rossola', 'Rotberger', 'RoterVeltliner', 'Rotgipfler', 'Rougeon', 'Roupeiro', 'Roussanne', 'RoyaldeAlloza', 'Rubin', 'Rubired', 'Ruché', 'Ruen', 'Rufete', 'Ruggine', 'Ruländer', 'Räuschling', 'Sabrevois', 'Sacy', 'Sagrantino', 'Samsó', 'Sangiovese', 'Saperavi', 'Sarba', 'SauvignonBlanc', 'SauvignonGris', 'SavagninBlanc', 'Savatiano', 'Scheurebe', 'Schiava', 'SchiavaGentile', 'SchiavaGrigia', 'Schioppettino', 'Schwarzriesling', 'Schönburger', 'Sciacarello', 'Sciascinoso', 'SearaNova', 'Segalin', 'Seibel', 'Sercial', 'Sercialinho', 'SeyvalBlanc', 'ShirokaMelnishka', 'Sibirkovi', 'Sideritis', 'Siegerrebe', 'Silvaner/Sylvaner', 'Smederevka', 'Solaris', 'Sousão', 'SouvignierGris', 'Spätburgunder', 'St.Croix', 'St.Laurent', 'Steuben', 'Sultana', 'Sultaniye', 'Sumoll', 'SumollBlanc', 'Susumaniello', 'SwensonWhite', 'Symphony', 'Syrah/Shiraz', 'Syriki', 'Szürkebarát', 'Sémillon', 'Síria', 'TamaioasaRomaneasca', 'Tamarez', 'Tannat', 'Tarrango', 'Tazzelenghe', 'Tempranillo', 'TempranilloBlanco', 'Teroldego', 'Terrano', 'Terrantez', 'Terret', 'Thrapsathiri', 'Tibouren', 'Timorasso', 'TintaAmarela', 'TintaBarroca', 'TintaCaiada', 'TintaCarvalha', 'TintaFrancisca', 'TintaMadeira', 'TintaMiúda', 'TintaNegraMole', 'TintaRoriz', 'TintadeToro', 'TintadelPais', 'Tintilia', 'Tintilla', 'TintoCão', 'TintoFino', 'TintoreDiTramonti', 'TocaiFriulano', 'TocaiItalico', 'Torbato', 'Torrontés', 'TourigaFranca', 'TourigaNacional', 'Trajadura', 'Traminer', 'Traminette', 'Trebbiano', 'TrebbianoGiallo', 'TrebbianoRomagnolo', 'TrebbianoToscano', 'Treixadura', 'Trepat', 'Trincadeira', 'Triomphe', 'Trollinger', 'Trousseau', 'TsimlyanskyCherny', 'Tsolikouri', 'Turan', 'Turbiana', 'UghettadiCanneto', 'UgniBlanc', 'UlldeLlebre', 'UvaRara', 'Vaccareze', 'Valdiguie', 'ValentinoNero', 'Verdeca', 'Verdejo', 'Verdelho', 'Verdello', 'Verdicchio', 'Verdiso', 'VerduzzoFriulano', 'Vermentino', 'VermentinoNero', 'Vernaccia', 'VernacciadiOristano', 'VernacciadiSanGimignano', 'Vernatsch', 'Vespaiola', 'Vespolina', 'VidalBlanc', 'Vidiano', 'ViendeNus', 'Vignoles', 'Vijiriega', 'Vilana', 'VillardNoir', 'Vincent', 'Vinhão', 'Viognier', 'Violeta', 'Viorica', 'Viosinho', 'Vital', 'Vitovska', 'Viura', 'Vranac', 'Weissburgunder', 'Welschriesling', 'Xarel-lo', 'Xinomavro', 'Xynisteri', 'Zalema', 'Zelen', 'Zengö', 'Zibibbo', 'Zierfandler', 'Zinfandel', 'ZinfandelWhite', 'Zlahtina', 'Zweigelt', 'Zéta', 'ÁguaSanta', 'Öküzgözü']
    # Convert grapes list to set for faster lookup
    grapes_set = set(grapes)

    # Create a DataFrame with all grape columns populated with 0
    zeros_df = pd.DataFrame(0, index=X_pred.index, columns=grapes_columns)

    # Set 1 for the columns that are in the grapes list
    for grape in grapes:
        if grape in grapes_columns:
            zeros_df[grape] = 1

    # Concatenate X_pred with zeros_df along the columns axis
    X_pred = pd.concat([X_pred, zeros_df], axis=1)

    return X_pred

def pred(model, X_pred: pd.DataFrame):

    '''
    Return a dictonary that provides the predicted wines based on the provided X_pred
    '''

    y_pred = model.predict(X_pred)
    y_pred = y_pred.toarray()

    foods = ['Appetizer', 'Beef', 'Cured Meat', 'Game Meat', 'Lamb', 'Pasta', 'Pork', 'Poultry',
             'Rich Fish', 'Shellfish', 'Veal', 'Vegetarian']
    foods_index = np.where(y_pred[0]==1)[0].tolist()
    foods_to_choose = []
    for i in foods_index:
        foods_to_choose.append(foods[i])

    return {"foods": foods_to_choose}
