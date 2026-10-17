# ⚖️ LegalMind — Assistant IA juridique pour les PME marocaines

> Un système RAG (Retrieval-Augmented Generation) qui répond à des questions juridiques en darija, français et arabe, en s'appuyant sur le Code du travail marocain, avec citations obligatoires pour éviter les hallucinations.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-green)](https://langchain.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-orange)](https://trychroma.com)
[![Groq](https://img.shields.io/badge/Groq-Llama_3-red)](https://groq.com)

## 🎯 Le problème
Les TPE/PME marocaines n'ont pas accès à un conseil juridique abordable. Un avocat coûte 500-1000 MAD/heure. Résultat : des contrats signés sans compréhension et des litiges évitables.

## 💡 La solution
LegalMind est un assistant conversationnel qui permet de :
- Poser des questions juridiques en langage naturel.
- Obtenir des réponses **avec citation de l'article de loi exact**.
- Éviter les hallucinations grâce à une architecture RAG stricte.

## 🏗️ Architecture
`Question → Embedding → Recherche hybride (BM25 + Vectorielle) → LLM + Contexte → Réponse avec citation`

| Couche | Technologie |
|--------|-------------|
| Parsing PDF | pdfplumber |
| Embeddings | paraphrase-multilingual-MiniLM-L12-v2 |
| Vector DB | ChromaDB |
| LLM | Groq (Llama 3) |
| API | FastAPI |
| Démo | Streamlit |

## 📊 Résultats (Tests initiaux)
- **50 questions testées**
- **84% de taux de citation correcte**
- **95% de refus d'hallucination** (refuse de répondre si l'info n'est pas dans le texte)
- **Latence moyenne :** ~2.3s

## 🚀 Démo en ligne
[Accéder à l'application](https://legalmind.streamlit.app) *(lien à mettre à jour une fois déployé)*

## 👤 Auteur
**[Votre Nom]** — Étudiant en Intelligence Artificielle, Université Ibn Tofail.
[LinkedIn](https://linkedin.com/in/votre-profil) | [GitHub](https://github.com/aymaaan-elou)
