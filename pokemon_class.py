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
        
        """
        Takes no parameter, accesses df in class variable pokemon_df, a preprocessed df
        """


        encoded_pokemons = self.pokemon_df.to_dummies(columns=["primary", "secondary", 'stage'])

        X = encoded_pokemons[:, 2:].to_numpy() #all rows, but starting from column 2: height_cm
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
        Method takes 5 parameters to classify pokemon and creates filler data for the 'pred_df', a dataframe used to one-hot-encode 
        all values in 'primary', 'secondary' and 'stage' features.
        
        Parameters go from:
        [height, weight, primary, secondary, stage]
        --> 5 parameters
        
        to:

        [height, weight, primary_1, primary_2... primary_n, secondary_1, secondary_2... secondary_n, stage_1, stage_2, stage_3]
        --> 36 parameters.

        Dataframe is then converted to numpy array.
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
        stage_fillers = [1,2,3]*10 + [1,2,3,1,2,3] # 3*10 + 6 = 36


        pred_df = pl.DataFrame({
            'height': [height] + h_fillers, 'weight': [weight] + w_fillers,
            'primary': [primary]+ p_fillers, 'secondary': [secondary] + s_fillers,
            'stage': [evo_stage] + stage_fillers
        })

        pred_df = pred_df.to_dummies(columns=["-primary", "secondary", 'stage']) #polars encode method, to_dummies()
        pred_array = pred_df[0,:].to_numpy()

        trained_classifier, trained_scaler = self.train_modell()

        scaled_array = trained_scaler.transform(pred_array)
        prediction = trained_classifier.predict(scaled_array)

        return prediction


    def pokemon_stats(self, pokemon):
        """
        Takes predicted pokemon as parameter, filters pokemon_df for stats.
        Extracts average of all training instances from the describe() and returns tuple of stats
        """
        
        stats_range = self.pokemon_df.filter(pl.col('name') == pokemon)    

        height_weight = stats_range.describe().row(2)[3:5]
        nr = stats_range[0,0]
        primary_type, secondary_type = stats_range[0, 5], stats_range[0, 4]

        stats = (nr,) + (pokemon,) + height_weight + (primary_type, secondary_type)
        
        return stats



