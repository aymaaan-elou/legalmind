import os
import streamlit as st
from groq import Groq
from src.retriever import recherche_hybride
import chromadb
from sentence_transformers import SentenceTransformer

# 1. Récupération sécurisée de la clé API
# On cherche d'abord dans les secrets Streamlit (Cloud), puis dans l'environnement (Local)
api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")

if not api_key:
    raise ValueError("Erreur : La clé GROQ_API_KEY est introuvable. Vérifiez les Secrets sur Streamlit Cloud.")

client_llm = Groq(api_key=api_key)

# 2. Initialisation des modèles et de la base de données
modele_embedding = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
client_chroma = chromadb.PersistentClient(path="./chroma_db")

# On récupère la collection. Si elle n'existe pas, get_or_create la crée (vide)
collection = client_chroma.get_or_create_collection(name="lois_maroc")

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
    # On récupère tous les textes stockés dans la base ChromaDB
    resultats_db = collection.get()
    chunks = resultats_db['documents']
    
    if not chunks:
        return "Erreur : La base de données est vide. Le fichier PDF n'a pas encore été traité. Veuillez patienter ou vérifier le fichier data/raw/."

    # Recherche des 5 passages les plus pertinents
    chunks_recuperes = recherche_hybride(question, chunks, modele_embedding, collection, k=5)
    
    # Construction du prompt et appel à l'IA
    prompt = construire_prompt(question, chunks_recuperes)
    
    reponse = client_llm.chat.completions.create(
        model="openai/gpt-oss-20b", 
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    return reponse.choices[0].message.content
