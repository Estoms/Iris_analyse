import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Titre de l'application
st.title("Exploration et Classification - Dataset IRIS 🌸")

# Charger la base de données
st.header("1. Chargement des données")
uploaded_file = st.file_uploader("Téléversez le fichier IRIS.csv", type=["csv"])
if uploaded_file:
    # Lire le fichier
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())
    
    # Afficher des statistiques
    st.write("**Statistiques descriptives :**")
    st.write(df.describe())

    # Visualisation des données
    st.header("2. Visualisation des données 📊")
    st.write("Sélectionnez deux colonnes pour tracer un graphique interactif.")
    
    col1 = st.selectbox("Axe X :", df.columns[:-1])
    col2 = st.selectbox("Axe Y :", df.columns[:-1])

    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x=col1, y=col2, hue="species", ax=ax)
    st.pyplot(fig)
    
    # Boxplot pour les distributions
    st.write("Distribution des colonnes par classe :")
    col_dist = st.selectbox("Choisissez une colonne :", df.columns[:-1])
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x="species", y=col_dist, ax=ax)
    st.pyplot(fig)

    # Modèle k-NN
    st.header("3. Modèle de classification - k-NN 🤖")
    st.write("Entraînons un modèle k-NN pour prédire les espèces.")

    # Préparation des données
    X = df.iloc[:, :-1]
    y = df["species"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Hyperparamètre k
    k = st.slider("Nombre de voisins (k) :", 1, 15, 3)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Prédiction et précision
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"**Précision du modèle :** {accuracy:.2f}")

    # Prédiction interactive
    st.subheader("Prédiction avec vos propres valeurs")
    sepal_length = st.number_input("Longueur du sépale :", value=5.0)
    sepal_width = st.number_input("Largeur du sépale :", value=3.5)
    petal_length = st.number_input("Longueur du pétale :", value=1.5)
    petal_width = st.number_input("Largeur du pétale :", value=0.2)
    
    if st.button("Prédire"):
        pred = knn.predict([[sepal_length, sepal_width, petal_length, petal_width]])
        st.write(f"**Espèce prédite :** {pred[0]}")
