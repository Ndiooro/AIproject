import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

st.title("Détection d'intrusions NBI-IoT")

uploaded_file = st.file_uploader("Téléversez un fichier CSV contenant le dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Aperçu du dataset :", df.head())

    if 'label' not in df.columns:
        st.error("Le fichier CSV doit contenir une colonne 'label'")
    else:
        # Préparation des données
        X = df.drop('label', axis=1)
        y = df['label']

        smote = SMOTE(random_state=42)
        X, y = smote.fit_resample(X, y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        models = {
            "Régression Logistique": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(),
            "SVM": SVC()
        }

        selected_model = st.selectbox("Choisissez un algorithme de classification :", list(models.keys()))

        if st.button("Entraîner et évaluer le modèle"):
            model = models[selected_model]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = f1_score(y_test, y_pred, average='binary')
            st.success(f"F1 Score obtenu avec {selected_model} : {score:.4f}")
