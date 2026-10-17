import os
from groq import Groq

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
    # Simulation de contexte pour tester l'API
    chunks_recuperes = ["Ceci est un texte de test pour le contexte."]
    
    prompt = construire_prompt(question, chunks_recuperes)
    reponse = client_llm.chat.completions.create(
        model="openai/gpt-oss-20b", 
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    return reponse.choices[0].message.content
