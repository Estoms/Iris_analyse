import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# === Application Titre ===
st.title("Analyse et Clustering des Données IRIS")
st.markdown("### By Estoms")

# === Chargement des Données ===
st.header("1. Chargement de la Base de Données")
file_path = "./data/IRIS.csv"
data = pd.read_csv(file_path)
st.dataframe(data)

# Aperçu des données
st.subheader("Aperçu des données :")
st.write(data.describe())

# === Option : Clustering avec K-Means ===
st.header("2. Clustering avec K-Means")
columns = st.multiselect("Sélectionnez les colonnes pour le clustering :", data.columns[:-1], default=data.columns[1:4])

if len(columns) < 2:
    st.error("Veuillez sélectionner au moins deux colonnes pour effectuer le clustering.")
else:
    # Normalisation des données
    scaler = StandardScaler()
    X = scaler.fit_transform(data[columns])

    # Sélection du nombre de clusters
    n_clusters = st.slider("Nombre de clusters (k)", min_value=2, max_value=10, value=3)

    # K-Means
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
    data['Cluster'] = kmeans.fit_predict(X)

    # Visualisation des clusters
    st.subheader("Visualisation des Clusters")
    fig, ax = plt.subplots()
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=data['Cluster'], palette='viridis', ax=ax)
    ax.set_title("Clusters avec K-Means")
    st.pyplot(fig)

# === Option : Classification avec KNN ===
st.header("3. Classification avec KNN")
X_knn = data.iloc[:, :-2].values  # Toutes les colonnes sauf les deux dernières
if 'species' in data.columns:
    y_knn = data['species'].values  # Cible

    # Division des données
    X_train, X_test, y_train, y_test = train_test_split(X_knn, y_knn, test_size=0.3, random_state=1)

    # KNN
    k = st.slider("Choisissez le nombre de voisins (k) pour KNN", min_value=1, max_value=10, value=5)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    # Résultats
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    st.write(f"### Précision du modèle : {accuracy:.2f}")
    st.write("### Matrice de Confusion :")
    st.dataframe(conf_matrix)

# === Footer ===
st.markdown("---")
st.markdown("### By Estoms")
