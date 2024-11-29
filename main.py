import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# 1. Charger la base de données
st.title("Exploration de la base IRIS 🌺")
uploaded_file = "IRIS.csv"  # Nom du fichier local

@st.cache
def load_data(file_path):
    return pd.read_csv(file_path)

data = load_data(uploaded_file)

# 2. Aperçu des données
st.subheader("Aperçu de la Base de Données")
if st.checkbox("Afficher les 10 premières lignes"):
    st.write(data.head(10))

st.subheader("Statistiques Descriptives")
if st.checkbox("Afficher les statistiques descriptives"):
    st.write(data.describe())

# 3. Graphiques interactifs
st.subheader("Visualisations")
graph_type = st.selectbox(
    "Choisissez le type de graphique", ["Boxplot", "Histogramme", "Scatter Plot"]
)

if graph_type == "Boxplot":
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=data, palette="pastel")
    st.pyplot(plt)

elif graph_type == "Histogramme":
    column = st.selectbox("Choisissez une colonne", data.columns[:-1])  # Hors 'species'
    plt.figure(figsize=(10, 5))
    sns.histplot(data[column], kde=True, color="blue")
    st.pyplot(plt)

elif graph_type == "Scatter Plot":
    x_col = st.selectbox("Axe X", data.columns[:-1])
    y_col = st.selectbox("Axe Y", data.columns[:-1])
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x=data[x_col], y=data[y_col], hue=data['species'], palette="deep")
    st.pyplot(plt)

# 4. Algorithme KNN
st.subheader("Prédictions avec KNN")

if st.checkbox("Lancer une classification KNN"):
    # Sélection des variables
    X = data.iloc[:, :-1]  # Toutes les colonnes sauf 'species'
    y = data['species']

    # Division en train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Entrée utilisateur pour KNN
    k = st.slider("Choisissez la valeur de K", 1, 15, 3)

    # Modèle
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Résultats
    st.text("Rapport de Classification")
    st.text(classification_report(y_test, y_pred))

    # Prédiction utilisateur
    st.subheader("Tester une nouvelle prédiction")
    inputs = [st.number_input(f"Valeur pour {col}", value=0.0) for col in X.columns]
    if st.button("Prédire"):
        result = model.predict([inputs])
        st.success(f"La classe prédite est : {result[0]}")
