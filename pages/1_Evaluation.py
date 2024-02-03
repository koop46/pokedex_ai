
import seaborn as sns 
import streamlit as st
import matplotlib.pyplot as plt 
from pokemon_class import Pokemon_modell
from sklearn.metrics import classification_report, confusion_matrix


modell = Pokemon_modell()
classifier, _, X_test, y_test = modell.train_modell()
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)


body = st.container()

with body:
    classification_report_tab, confusion_matrix_tab = st.tabs(['Classification Report', 'Confusion Matrix'])

    with classification_report_tab:
        st.text("""First column is name. Second is 'precision': % of true positives. 
Third is 'recall': % of positive samples found. 
Fourt is 'f1-score': overall performance of model.
Last one is amount of instances in test set
""")
        st.text(" ")    
        st.text(classification_report(y_test, y_pred))

    with confusion_matrix_tab:
        fig = plt.figure(figsize=(50, 50)) # Size in inches
        sns.heatmap(cm, annot=True)

        plt.title("Confusion matrix")
        plt.ylabel('Truth')
        plt.xlabel('Prediction')
        st.pyplot(fig)


