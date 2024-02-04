import streamlit as st 
from piclinks import poke_pics
from pokemon_class import Pokemon_modell


# - instantiate modell, df, classifier and scaler
# - body:
#    -col1:
#        input form with number_inputs, selectionboxes & radiobuttons
#    -col2:
    #     just for spacing
    # -col3:
    #     where predictions is presented, with name, picture & data


modell = Pokemon_modell()
df = modell.pokemon_df
classifier, scaler, _, _ = modell.train_modell() #we're only going to use classifier and scaler. X_test and y_test are return for the evaluation

body = st.container()

with body:
    col1, col2, col3 = st.columns([.4, .2, .4]) # ratios for columns 40%, 20% & 20%s

    with col1:

        form = st.form("input_form")
        form.subheader("Input stats")

        height = form.number_input('Height (cm)',
                                    min_value=15.0,
                                    max_value=890.0, 
                                    step=0.5, 
                                    value=70.0, 
                                    placeholder='Enter centimeters', 
                                    format='%f'
                                    )
        weight = form.number_input('Weight (kg)', 
                                   min_value=3.0, 
                                   max_value=200.0, 
                                   step=0.5, 
                                   value=8.0, 
                                   placeholder='Enter kilograms', 
                                   format='%f'
                                   )

        primary_type = form.selectbox('Primary type', 
                                      ('Grass', 'Rock', 'Fighting', 
                                       'Bug', 'Fairy', 'Ice', 
                                       'Water', 'Ghost', 'Psychic', 
                                       'Dragon', 'Ground', 'Fire', 
                                       'Electric', 'Normal', 'Poison')
                                       )
        primary_type = primary_type.lower()
        secondary_type = form.selectbox('Secondary type', 
                                        ('Normal', 'Rock', 'Water', 
                                         'Steel', 'Flying', 'Ice', 
                                         'Fighting', 'Electric', 'Poison', 
                                         'Bug', 'Fairy', 'Dragon', 
                                         'Psychic', 'Ground', 'Grass', 'Fire')
                                         )
        secondary_type = secondary_type.lower()


        evolutionary_stage = form.radio('Evolutionary stage', 
                                        [1, 2, 3],
                                        horizontal=True
                                        )
        
        if form.form_submit_button("Search"):
            pass

    with col2:
        pass

    with col3:

        predicted_pokemon = modell.predict_pokemon(height, weight, primary_type, secondary_type, evolutionary_stage)
        pokemon = predicted_pokemon[0].title()

        nr_stat, name_stat, height_stat, weight_stat, primary_stat, secondary_stat = modell.pokemon_stats(pokemon.lower())

        st.subheader(f"Predicted pokemon: #{nr_stat} | {pokemon}")
        st.image(poke_pics[pokemon][1])
        
        with st.expander(f"Pokemon stats"):

            st.text(f"""  
Avg height: {height_stat:.1f} cm 
Avg weight: {weight_stat:.1f} kg

Primary type: {primary_stat.capitalize()}
Secondary type: {secondary_stat.capitalize()}

Evolutionary stage: {evolutionary_stage}/3
""")


