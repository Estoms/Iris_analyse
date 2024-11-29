import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# Titre de l'application
st.title("Exploration et Prédiction sur la Base IRIS 🌼")

# Chemin vers le fichier de données
DATA_PATH = "data/IRIS.csv"

# Fonction pour charger les données
@st.cache_data
def load_data(file_path):
    """Charge les données depuis un fichier CSV."""
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        st.error("Erreur : Le fichier de données est introuvable.")
        return None
    except pd.errors.ParserError:
        st.error("Erreur : Problème lors de la lecture du fichier CSV.")
        return None

# Charger les données
data = load_data(DATA_PATH)

if data is not None:
    # Section 1 : Exploration des données
    st.header("1️⃣ Exploration des Données")
    st.subheader("Aperçu des Données")
    st.write(data.head())

    st.subheader("Statistiques Descriptives")
    st.write(data.describe())

    # Graphiques interactifs
    st.subheader("Visualisation des Données")
    graph_type = st.radio(
        "Choisissez le type de graphique :",
        ("Boxplot", "Histogramme", "Scatter Plot")
    )

    if graph_type == "Boxplot":
        st.write("Boxplot des caractéristiques :")
        plt.figure(figsize=(10, 5))
        sns.boxplot(data=data, palette="pastel")
        st.pyplot(plt)
        plt.clf()  # Nettoyer la figure pour éviter des chevauchements

    elif graph_type == "Histogramme":
        column = st.selectbox("Choisissez une colonne :", data.columns[:-1])
        st.write(f"Histogramme de la colonne {column} :")
        plt.figure(figsize=(10, 5))
        sns.histplot(data[column], kde=True, color="blue")
        st.pyplot(plt)
        plt.clf()

    elif graph_type == "Scatter Plot":
        x_col = st.selectbox("Axe X :", data.columns[:-1])
        y_col = st.selectbox("Axe Y :", data.columns[:-1])
        st.write(f"Scatter Plot entre {x_col} et {y_col} :")
        plt.figure(figsize=(10, 5))
        sns.scatterplot(x=data[x_col], y=data[y_col], hue=data['species'], palette="deep")
        st.pyplot(plt)
        plt.clf()

    # Section 2 : Prédictions avec KNN
    st.header("2️⃣ Prédiction avec l'Algorithme KNN")
    if st.checkbox("Activer la Prédiction KNN"):

        # Préparer les données pour KNN
        X = data.iloc[:, :-1]  # Caractéristiques
        y = data['species']  # Classe cible

        # Division en jeu d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Choix de la valeur de K
        k = st.slider("Choisissez la valeur de K :", 1, 15, 3)

        # Création du modèle
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Afficher les résultats
        st.subheader("Rapport de Classification")
        st.text(classification_report(y_test, y_pred))

        # Prédiction utilisateur
        st.subheader("Tester une Nouvelle Observation")
        inputs = []
        for col in X.columns:
            value = st.number_input(f"Valeur pour {col}", value=0.0)
            inputs.append(value)

        if st.button("Prédire"):
            try:
                result = model.predict([inputs])
                st.success(f"La classe prédite est : {result[0]}")
            except ValueError:
                st.error("Erreur : Veuillez entrer des valeurs valides pour toutes les colonnes.")

# Footer
st.write("---")
st.write("Application créée avec ❤️ par [Votre Nom].")
