import streamlit as st
# from src.llm import repondre

st.set_page_config(page_title="LegalMind", page_icon="⚖️")
st.title("⚖️ LegalMind — Assistant juridique")
st.caption("Posez une question sur le Code du travail marocain")

question = st.text_input("Votre question :", placeholder="Ex: Quelle est la durée du préavis ?")

if st.button("Demander à l'IA") and question:
    with st.spinner("Recherche dans les textes de loi..."):
        # reponse = repondre(question)
        reponse = "Ceci est une réponse simulée pour tester l'interface."
    st.markdown("### Réponse :")
    st.success(reponse)
    st.warning("⚠️ Projet académique — Ne remplace pas un avocat.")
