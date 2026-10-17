from fastapi import FastAPI
from pydantic import BaseModel
# from src.llm import repondre

app = FastAPI(title="LegalMind API")

class Question(BaseModel):
    texte: str

@app.post("/chat")
async def chat(q: Question):
    # return {"reponse": repondre(q.texte)}
    return {"reponse": "L'IA est en cours de connexion..."}

@app.get("/health")
async def health():
    return {"status": "ok"}
