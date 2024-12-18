import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# === Application Header ===
st.title("Exploration des Données IRIS avec Estoms")
st.markdown("#### Une application interactive pour analyser et visualiser les données IRIS")

# === Chargement des Données ===
st.header("1. Importer les Données")
data_path = "./data/IRIS.csv"
iris_data = pd.read_csv(data_path)
st.write("Voici un échantillon des données chargées :")
st.dataframe(iris_data.head())

# Aperçu des statistiques
st.subheader("Statistiques des Données :")
st.write(iris_data.describe())

# === Analyse : Clustering avec K-Means ===
st.header("2. Clustering des Fleurs avec K-Means")
available_columns = st.multiselect("Colonnes disponibles pour le clustering :", iris_data.columns[:-1], default=iris_data.columns[1:4])

if len(available_columns) >= 2:
    # Normalisation des données
    normalizer = MinMaxScaler()
    normalized_data = normalizer.fit_transform(iris_data[available_columns])

    # Paramètres de clustering
    num_clusters = st.slider("Nombre de groupes à créer :", min_value=2, max_value=8, value=3)

    # Exécution de K-Means
    kmeans_model = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans_model.fit_predict(normalized_data)
    iris_data['Groupes'] = cluster_labels

    # Visualisation des groupes
    st.subheader("Visualisation des Groupes")
    fig, ax = plt.subplots()
    sns.scatterplot(x=normalized_data[:, 0], y=normalized_data[:, 1], hue=iris_data['Groupes'], palette="cool", ax=ax)
    ax.set_title("Clusters Générés par K-Means")
    st.pyplot(fig)

    # Affichage des centres des clusters
    st.subheader("Centres des Clusters")
    cluster_centers = kmeans_model.cluster_centers_
    st.write(pd.DataFrame(cluster_centers, columns=available_columns))
else:
    st.warning("Veuillez sélectionner au moins deux colonnes pour effectuer le clustering.")

# === Analyse : Classification avec KNN ===
st.header("3. Prédiction des Espèces avec KNN")

if 'species' in iris_data.columns:
    # Préparation des données
    X_features = iris_data.select_dtypes(include=['number']).iloc[:, :-1].values  # Colonnes numériques
    y_target = iris_data['species'].values

    # Division des jeux de données
    X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.3, random_state=7)

    # Paramètre du modèle KNN
    k_neighbors = st.slider("Nombre de voisins pour KNN :", min_value=1, max_value=15, value=5)
    knn_model = KNeighborsClassifier(n_neighbors=k_neighbors)

    # Entraînement et Prédiction
    knn_model.fit(X_train, y_train)
    predictions = knn_model.predict(X_test)

    # Résultats
    model_accuracy = accuracy_score(y_test, predictions)
    confusion_mat = confusion_matrix(y_test, predictions)

    st.write(f"### Précision du Modèle : {model_accuracy:.2f}")
    st.write("### Matrice de Confusion :")
    st.dataframe(confusion_mat)

    # Rapport de classification
    st.subheader("Rapport de Classification")
    classification_rep = classification_report(y_test, predictions, output_dict=True)
    st.dataframe(pd.DataFrame(classification_rep).transpose())

# === Nouvelle Fonctionnalité : Analyse des Corrélations ===
st.header("4. Analyse des Corrélations")
correlation_method = st.radio("Choisissez la méthode de corrélation :", ["pearson", "spearman", "kendall"])
st.subheader("Matrice de Corrélation")

# Filtrer les colonnes numériques pour éviter les erreurs
numerical_data = iris_data.select_dtypes(include=['number'])

if not numerical_data.empty:
    correlation_matrix = numerical_data.corr(method=correlation_method)
    fig, ax = plt.subplots()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Matrice de Corrélation des Attributs Numériques")
    st.pyplot(fig)
else:
    st.warning("Aucune colonne numérique trouvée pour calculer la matrice de corrélation.")

# === Footer ===
st.markdown("---")
st.markdown("#### by Estoms")
