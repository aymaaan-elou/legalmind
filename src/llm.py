import os
import streamlit as st
from groq import Groq
from src.retriever import recherche_hybride
import chromadb
from sentence_transformers import SentenceTransformer

# --- INITIALISATION DE LA CLÉ API ---
# On lit la clé directement dans les secrets de Streamlit.
# C'est la méthode la plus fiable sur Streamlit Cloud.
api_key = st.secrets["GROQ_API_KEY"]

# Initialisation du client Groq avec cette clé.
client_llm = Groq(api_key=api_key)

# --- INITIALISATION DES MODÈLES ---
@st.cache_resource
def charger_modeles():
    modele_embedding = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    client_chroma = chromadb.PersistentClient(path="./chroma_db")
    collection = client_chroma.get_or_create_collection(name="lois_maroc")
    return modele_embedding, collection

modele_embedding, collection = charger_modeles()

def construire_prompt(question, chunks_recuperes):
    contexte = "\n\n".join([
        f"[Source: Code du travail, chunk {i}]\n{chunk}"
        for i, chunk in enumerate(chunks_recuperes)
    ])
    return f"""Tu es un assistant juridique marocain. Réponds UNIQUEMENT avec le contexte.
RÈGLES:
1. Si la réponse n'est pas dans le contexte, dis "Je ne trouve pas cette information."
2. Cite TOUJOURS l'article de loi.
CONTEXTE:
{contexte}
QUESTION: {question}
RÉPONSE:"""

def repondre(question):
    resultats_db = collection.get()
    chunks = resultats_db['documents']
    
    if not chunks:
        return "Erreur : La base de données est vide. Le fichier PDF n'a pas encore été traité. Veuillez patienter."

    chunks_recuperes = recherche_hybride(question, chunks, modele_embedding, collection, k=5)
    prompt = construire_prompt(question, chunks_recuperes)
    
    reponse = client_llm.chat.completions.create(
        model="openai/gpt-oss-20b", 
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    return reponse.choices[0].message.content
