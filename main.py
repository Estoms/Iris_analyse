import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Charger les données
data = pd.read_csv("IRIS.csv")

# Titre
st.title("Visualisation des Données")

# Type de graphique
graph_type = st.radio("Choisissez le type de graphique :", ["Boxplot", "Histogramme", "Scatter Plot"])

# Sélecteurs pour les axes X et Y
columns = data.columns.tolist()
x_col = st.selectbox("Axe X :", columns)
y_col = st.selectbox("Axe Y :", columns)

# Génération du graphique
if graph_type == "Scatter Plot":
    st.subheader(f"Scatter Plot entre {x_col} et {y_col} :")
    fig, ax = plt.subplots()
    sns.scatterplot(x=data[x_col], y=data[y_col], hue=data['species'], palette="viridis", ax=ax)
    st.pyplot(fig)
elif graph_type == "Boxplot":
    st.subheader(f"Boxplot de {y_col} par {x_col} :")
    fig, ax = plt.subplots()
    sns.boxplot(x=data[x_col], y=data[y_col], hue=data['species'], ax=ax)
    st.pyplot(fig)
elif graph_type == "Histogramme":
    st.subheader(f"Histogramme de {x_col} :")
    fig, ax = plt.subplots()
    sns.histplot(data[x_col], kde=True, color="blue", ax=ax)
    st.pyplot(fig)
