import streamlit as st
import joblib
import pandas as pd

# Charger le modèle
model = joblib.load('meilleur_modele_titanic.pkl')
features = joblib.load('feature_names.pkl')

st.title("🔮 Prédiction de survie - Titanic")
st.write("Entrez les informations du passager :")

# Formulaire
col1, col2 = st.columns(2)
with col1:
    pclass = st.selectbox("Classe", [1, 2, 3])
    sex = st.selectbox("Sexe", ["female", "male"])
    age = st.number_input("Âge", min_value=0, max_value=100, value=25)
with col2:
    sibsp = st.number_input("Frères/Soeurs/Conjoint", min_value=0, max_value=10, value=0)
    parch = st.number_input("Parents/Enfants", min_value=0, max_value=10, value=0)
    fare = st.number_input("Prix du billet", min_value=0.0, value=50.0)
    embarked = st.selectbox("Port d'embarquement", ["S", "C", "Q"])

if st.button("Prédire la survie"):
    # Préparation
    data = [[pclass, 1 if sex == "female" else 0, age, sibsp, parch, fare, 
             0 if embarked == "S" else 1 if embarked == "C" else 2]]
    
    # Prédiction
    proba = model.predict_proba(data)[0]
    prediction = model.predict(data)[0]
    
    # Affichage
    if prediction == 1:
        st.success(f"✅ Survie probable : {proba[1]:.1%}")
        st.balloons()
    else:
        st.error(f"❌ Mort probable : {proba[0]:.1%}")
