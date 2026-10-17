import streamlit as st
from src.llm import repondre

st.set_page_config(page_title="LegalMind", page_icon="⚖️", layout="centered")

st.title("⚖️ LegalMind — Assistant juridique")
st.caption("Posez une question sur le Code du travail marocain (Projet Académique)")

question = st.text_input("Votre question :", placeholder="Ex: Quelle est la durée du congé de maternité ?")

if st.button("Demander à l'IA") and question:
    with st.spinner("Recherche dans les textes de loi en cours..."):
        try:
            # Appel de la vraie fonction IA
            reponse = repondre(question)
            st.markdown("### Réponse :")
            st.success(reponse)
        except Exception as e:
            st.error(f"Une erreur est survenue : {e}")
            
    st.warning("⚠️ Ceci est un projet académique. Les réponses générées par l'IA ne remplacent en aucun cas l'avis d'un avocat.")
