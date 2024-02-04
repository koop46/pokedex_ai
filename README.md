# pokedex_ai
My first ml end-to-end ml project: a pokedex to classify pokemons

Libraries:
- streamlit
- scikit learn
- polars (instead of pandas)
- dataclasses
- Numpy

algoritm:
- knn classifier(10 neighbours)


The data was gathered from Bulbagarden.nets [Bulbapedia section](https://bulbapedia.bulbagarden.net/wiki/List_of_Pokémon_by_National_Pokédex_number#Generation_I).
I asked chatgpt to write me a function that generates data, (mock_data() in the pokedex.ipynb). From each individual pokemon I created 151 variations, averaging out to the weight and height from bulbapedia, a total of 22,801 rows with 7 columns. Datacleaning was therefore minimal, although I had to rerun the mock_data() function until it only generated positive values before exporting them as a csv file with polars.
I also collected links to a picture of each pokemon (picklinks.py).  

Once I had the csv file I loaded it onto a polars dataqframe, converted column datatypes to optimize memory usage and one-hot encoded three parameters (primary type, secondary type & evolutionary stage). Then I split the data into training & test data, transformed the data with sklearns StandardScaler before finally training the modell.

I evaluated accuracy with the help of confusion matrix and classification report. Next I created a class to organize the code and started working on the frontend.

With the streamlit library I built a simple frontend consisting of 2x numeric inputs for weight and height, 2x selection boxes for primary & secondary types and three radiobuttons for evolutionary stage.  
The app is multipage so under "Evaluation" the modell performance is presented in a classification report.










