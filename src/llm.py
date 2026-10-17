import os
import streamlit as st
from groq import Groq
from src.retriever import recherche_hybride
import chromadb
from sentence_transformers import SentenceTransformer

# 1. Récupération sécurisée de la clé API
api_key = None
try:
    # On cherche d'abord dans les secrets Streamlit (Cloud)
    if "GROQ_API_KEY" in st.secrets:
        api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

# Si non trouvé, on cherche dans les variables d'environnement (Local)
if not api_key:
    api_key = os.environ.get("GROQ_API_KEY")

# Si la clé est introuvable, on affiche une erreur propre dans l'interface
if not api_key:
    st.error("Erreur critique : La clé 'GROQ_API_KEY' est introuvable. Veuillez l'ajouter dans les Secrets de Streamlit Cloud.")
    st.stop() # Stoppe l'application proprement ici

# Initialisation du client Groq
client_llm = Groq(api_key=api_key)

# 2. Initialisation des modèles (mis en cache pour ne pas ralentir l'app)
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
    # Récupération des textes depuis ChromaDB
    resultats_db = collection.get()
    chunks = resultats_db['documents']
    
    if not chunks:
        return "Erreur : La base de données est vide. Le fichier PDF n'a pas encore été traité. Veuillez patienter quelques instants et rafraîchir la page."

    # Recherche des passages pertinents
    chunks_recuperes = recherche_hybride(question, chunks, modele_embedding, collection, k=5)
    
    # Construction du prompt et appel à l'IA
    prompt = construire_prompt(question, chunks_recuperes)
    
    reponse = client_llm.chat.completions.create(
        model="openai/gpt-oss-20b", 
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    return reponse.choices[0].message.content
