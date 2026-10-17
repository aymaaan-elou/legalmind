import streamlit as st
import os
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from sentence_transformers import SentenceTransformer
from src.llm import repondre

# --- Configuration de la page ---
st.set_page_config(page_title="LegalMind", page_icon="⚖️", layout="centered")

# --- Fonction pour préparer la base de données si elle n'existe pas ---
@st.cache_resource(show_spinner=False)
def initialiser_base_de_donnees():
    chemin_pdf = "data/raw/code_travail_maroc.pdf"
    chemin_db = "./chroma_db"

    # On vérifie si la base existe déjà
    if not os.path.exists(chemin_db):
        if not os.path.exists(chemin_pdf):
            return "Erreur : Le fichier PDF est introuvable dans data/raw/"
        
        # 1. Extraction du texte
        with pdfplumber.open(chemin_pdf) as pdf:
            pages = [page.extract_text() for page in pdf.pages if page.extract_text()]
        
        texte_complet = "\n\n".join(pages)
        
        # 2. Découpage en chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(texte_complet)
        
        # 3. Création des embeddings
        modele_embedding = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        vecteurs = modele_embedding.encode(chunks, show_progress_bar=False)
        
        # 4. Stockage dans ChromaDB
        client = chromadb.PersistentClient(path=chemin_db)
        collection = client.get_or_create_collection(name="lois_maroc")
        collection.add(
            documents=chunks,
            embeddings=vecteurs.tolist(),
            ids=[f"chunk_{i}" for i in range(len(chunks))]
        )
        return "Base de données créée avec succès !"
    
    return "Base de données déjà existante."

# --- Interface Utilisateur ---
st.title("⚖️ LegalMind — Assistant juridique")
st.caption("Posez une question sur le Code du travail marocain (Projet Académique)")

# Initialisation au démarrage de l'app
with st.spinner("Initialisation du système (peut prendre 1 minute au premier lancement)..."):
    message_init = initialiser_base_de_donnees()
    if "Erreur" in message_init:
        st.error(message_init)
        st.stop() # On arrête l'application si le PDF est manquant

# Zone de recherche
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
