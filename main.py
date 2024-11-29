import streamlit as st

# Titre de l'application
st.title("Vérification de Streamlit")

# Champ texte pour entrer un nom
nom = st.text_input("Entrez votre nom :")

# Bouton pour valider
if st.button("Valider"):
    if nom:
        st.success(f"Bonjour, {nom} ! 🎉 Streamlit fonctionne correctement.")
    else:
        st.warning("Veuillez entrer votre nom pour continuer.")
