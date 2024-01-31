import streamlit as st 
from piclinks import poke_pics
from pokemon_class import Pokemon_modell

modell = Pokemon_modell()
df = modell.pokemon_df
classifier, scaler = modell.train_modell()

body = st.container()

with body:
    col1, col2, col3 = st.columns([.4, .2, .4])

    with col1:

        form = st.form("input_form")
        form.subheader("Input stats")

        height = form.number_input('Height (cm)', min_value=50.0, max_value=400.0, step=0.5, value=100.0, placeholder='Enter centimeters', format='%f')
        weight = form.number_input('Weight (kg)', min_value=3.0, max_value=200.0, step=0.5, value=75.5, placeholder='Enter kilograms', format='%f')

        primary_type = form.selectbox('Primary type', ('Water', 'Rock', 'Fighting', 'Bug', 'Fairy', 'Ice', 'Grass', 'Ghost', 'Psychic', 'Dragon', 'Ground', 'Fire', 'Electric', 'Normal', 'Poison'))
        primary_type = primary_type.lower()


        secondary_type = form.selectbox('Secondary type', ('Normal', 'Rock', 'Water', 'Steel', 'Flying', 'Ice', 'Fighting', 'Electric', 'Poison', 'Bug', 'Fairy', 'Dragon', 'Psychic', 'Ground', 'Grass', 'Fire'))
        secondary_type = secondary_type.lower()


        evolutionary_stage = form.radio(
            'Evolutionary stage', 
            [1, 2, 3],
            horizontal=True
        )
        if form.form_submit_button("Search"):
            pass
    with col2:
        pass

    with col3:

        predicted_pokemon = modell.predict_pokemon(height, weight, primary_type, secondary_type, evolutionary_stage)
        pokemon = predicted_pokemon[0].capitalize()
        stats = modell.pokemon_stats(pokemon.lower())

        st.subheader(f"Predicted pokemon: {pokemon}")
        st.image(poke_pics[pokemon][1])
        
        with st.expander(f"#{stats[0]} | {stats[1].capitalize()}"):

            st.text(f"""  
Avg height: {stats[2]} cm 
Avg weight: {stats[3]} kg

Primary type: {stats[4].capitalize()}
Secondary type: {stats[5].capitalize()}

Evolutionary stage: {evolutionary_stage}/3
""")


