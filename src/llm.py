import os
from groq import Groq
from src.retriever import recherche_hybride
import chromadb
from sentence_transformers import SentenceTransformer

# 1. Initialisation des modèles et de la base de données
# (On les charge une seule fois au démarrage pour la performance)
modele_embedding = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
client_chroma = chromadb.PersistentClient(path="./chroma_db")
collection = client_chroma.get_or_create_collection(name="lois_maroc")

# 2. Connexion à Groq (utilise le Secret Streamlit ou une variable d'environnement)
client_llm = Groq(api_key=os.environ.get("GROQ_API_KEY"))

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
    # On récupère les textes (attention : on doit passer 'chunks', on va le charger depuis ChromaDB)
    # Pour faire simple, on récupère tous les documents de ChromaDB
    resultats_db = collection.get()
    chunks = resultats_db['documents']
    
    if not chunks:
        return "Erreur : La base de données est vide. Le PDF n'a pas été traité."

    chunks_recuperes = recherche_hybride(question, chunks, modele_embedding, collection, k=5)
    prompt = construire_prompt(question, chunks_recuperes)
    
    reponse = client_llm.chat.completions.create(
        model="openai/gpt-oss-20b", 
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    return reponse.choices[0].message.content
