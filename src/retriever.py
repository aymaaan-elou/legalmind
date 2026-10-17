import numpy as np
from rank_bm25 import BM25Okapi

def recherche_hybride(question, chunks, modele_embedding, collection, k=5):
    tokenized_chunks = [chunk.split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    
    q_embedding = modele_embedding.encode([question])
    resultats_vecteur = collection.query(query_embeddings=q_embedding.tolist(), n_results=k)
    docs_vecteur = resultats_vecteur['documents'][0]
    
    scores_bm25 = bm25.get_scores(question.split())
    top_bm25_idx = np.argsort(scores_bm25)[-k:][::-1]
    docs_bm25 = [chunks[i] for i in top_bm25_idx]
    
    scores = {}
    for rang, doc in enumerate(docs_vecteur):
        scores[doc] = scores.get(doc, 0) + 1 / (rang + 1)
    for rang, doc in enumerate(docs_bm25):
        scores[doc] = scores.get(doc, 0) + 1 / (rang + 1)
    
    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:k]
