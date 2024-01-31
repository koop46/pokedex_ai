# pokedex_ai
My first ml end-to-end ml project: a pokedex to classify pokemons

Libraries:
- streamlit
- scikit learn
- polars
- dataclasses
- Numpy

algoritm:
- knn classifier(10 neighbours)


The data was gathered from Bulbagarden.nets [Bulbapedia section](https://bulbapedia.bulbagarden.net/wiki/List_of_Pokémon_by_National_Pokédex_number#Generation_I).
From every individual row I created 151 variations of mock data , that together averaged out to the original data from bulbapedia.
151 x 151 equaled to 22,801 rows with 7 columns. Datacleaning therefore minimal. Then the dataframe was exported as a csv file with polars library.
I also collected links to a picture of each pokemon that I saved in a gspreadsheet.

Once I had the csv file I loaded it onto a polars dataqframe, converted column datatypes to optimize memory usage and one-hot encoded three parameters (primary type, secondary type & evolutionary stage). Then I split the data into training & test data, transformed the data with sklearns StandardScaler before finally training the modell.

I evaluated accuracy with the help of confusion matrix.
Next I created a class and started working on the frontend.

With the streamlit library I built a simple frontend consisting of 2x numeric inputs for weight and height, 2x selection boxes for primary & secondary types and three radiobuttons for evolutionary stage.










