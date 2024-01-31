from dataclasses import dataclass
import polars as pl
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split



@dataclass
class Pokemon_modell:
    """
    Pokemon_modell class contains 1 df and 3 methods:
    - dataframe of all pokemons loaded from the 151_x_151_pokemons.csv file,
    - train_modell method that trains the modell with dataframe values
    - predict_pokemon that does the actual prediction
    - pokemons_stats to present data
    """


    pokemon_df = pl.read_csv('151_x_151_pokemons.csv').with_columns(

        pl.col("nr").cast(pl.UInt8),
        pl.col("name").cast(pl.Utf8),
        pl.col("height_cm").cast(pl.Float32),
        pl.col("weight_kg").cast(pl.Float32),
        pl.col("primary").cast(pl.Utf8),
        pl.col("secondary").cast(pl.Utf8),
        pl.col("stage").cast(pl.UInt8),

    )


    def train_modell(self):


        encoded_pokemons = self.pokemon_df.to_dummies(columns=["primary", "secondary", 'stage'])

        X = encoded_pokemons[:, 2:].to_numpy() #all rows, but starting from col. index 2: height_cm
        y = encoded_pokemons['name'].to_numpy() #name col. as target


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

        scaler = StandardScaler()
        scaler.fit(X_train)


        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)


        classifier = KNeighborsClassifier(n_neighbors=10)
        classifier.fit(X_train, y_train)

        return classifier, scaler


    def predict_pokemon(self, height, weight, primary, secondary, evo_stage):

        """ 

        The predict_pokemon function starts by creating filler data for the 'pred_df', 
        a dataframe used to one-hot-encode all values in 'primary', 'secondary' and 'stage' features.

        """

        p_fillers = [
            'psychic', 'bug', 'normal', 'fairy', 'dragon', 'electric', 'fighting', 'ice', 'poison', 'rock', 'ghost', 
            'ground', 'water',  'grass', 'fire', 'psychic', 'bug', 'normal', 'fairy', 'dragon', 'electric', 'fighting', 
            'ice', 'poison', 'rock', 'ghost', 'ground', 'water',  'grass', 'fire', 'ground', 'water',  'grass', 'fire', 'grass', 'fire'
        ]
        s_fillers = [
            'fighting',  'fairy', 'electric', 'ice', 'grass', 'psychic', 'flying', 'steel', 'normal', 'rock', 'bug', 'dragon',
            'water',  'ground', 'poison', 'fire', 'fighting',  'fairy', 'electric', 'ice', 'grass', 'psychic', 'flying', 'steel', 
            'normal', 'rock', 'bug', 'dragon', 'water',  'ground', 'poison', 'fire', 'water',  'ground', 'poison', 'fire'
            ] 


        w_fillers = [123]*36 #fill column length for encoding
        h_fillers = [321]*36
        stage_fillers = [1,2,3]*10 + [1,2,3,1,2,3]


        pred_df = pl.DataFrame({
            'height': [height] + h_fillers, 'weight': [weight] + w_fillers,
            'primary': [primary]+ p_fillers, 'secondary': [secondary] + s_fillers,
            'stage': [evo_stage] + stage_fillers
        })

        pred_df = pred_df.to_dummies(columns=["primary", "secondary", 'stage']) #encode
        pred_array = pred_df[0,:].to_numpy()

        trained_classifier, trained_scaler = self.train_modell()

        z_test = trained_scaler.transform(pred_array)
        prediction = trained_classifier.predict(z_test)

        return prediction


    def pokemon_stats(self, pokemon):
        
        stats_range = self.pokemon_df.filter(pl.col('name') == pokemon)    

        height_weight = stats_range.describe().row(2)[3:5]
        nr = stats_range[0,0]
        primary_type, secondary_type = stats_range[0, 5], stats_range[0, 4]

        stats = (nr,) + (pokemon,) + height_weight + (primary_type, secondary_type)
        
        return stats






po = Pokemon_modell()
classifier, scaler = po.train_modell()

# pokemon = po.predict_pokemon(166, 70, 'normal', 'normal', 3)

# print(pokemon)