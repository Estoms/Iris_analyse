import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Titre de l'application
st.title("Analyse des Données IRIS")
st.markdown("### By 3Stoms")

# Chargement des données
st.header("1. Chargement de la Base de Données")
file_path = "./data/IRIS.csv"
data = pd.read_csv(file_path)
st.dataframe(data)

# Description des données
st.subheader("Aperçu des données :")
st.write(data.describe())

# Manipulation des données (KNN)
st.header("2. Manipulation des Données avec KNN")

# Préparation des données
X = data.iloc[:, :-1].values  # Toutes les colonnes sauf la dernière
y = data.iloc[:, -1].values  # Dernière colonne

# Séparation en jeu d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# KNN
k = st.slider("Choisissez le nombre de voisins (k) pour KNN", min_value=1, max_value=10, value=5)
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Résultats du modèle
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
st.write(f"### Précision du modèle : {accuracy:.2f}")
st.write("### Matrice de Confusion :")
st.dataframe(conf_matrix)

# Visualisations des données
st.header("3. Visualisation des Données")

# Distribution des classes
st.subheader("Distribution des classes dans IRIS")
fig, ax = plt.subplots()
data['species'].value_counts().plot(kind='bar', color=['blue', 'green', 'orange'], ax=ax)
ax.set_title("Distribution des Classes")
st.pyplot(fig)

# Scatterplot
st.subheader("Relations entre les dimensions des fleurs")
fig, ax = plt.subplots()
sns.scatterplot(x=data['sepal_length'], y=data['sepal_width'], hue=data['species'], palette='muted', ax=ax)
ax.set_title("Scatterplot des dimensions des sépales")
st.pyplot(fig)

# Deuxième manipulation des données : Clustering
st.header("4. Clustering avec K-Means")

# K-Means
num_clusters = st.slider("Nombre de clusters pour K-Means", min_value=2, max_value=5, value=3)
kmeans = KMeans(n_clusters=num_clusters, random_state=1)
clusters = kmeans.fit_predict(X)

data['Cluster'] = clusters

st.subheader("Résultats du Clustering :")
st.write(data[['species', 'Cluster']].head())

# Visualisation des clusters
fig, ax = plt.subplots()
sns.scatterplot(x=data['sepal_length'], y=data['sepal_width'], hue=data['Cluster'], palette='deep', ax=ax)
ax.set_title("Clusters K-Means")
st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("### By 3Stoms")
