from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"mensagem": "Ola, Carlos! Sua primeira API esta no ar."}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/saudacao/{nome}")
def saudacao(nome: str):
    return {"mensagem": f"Ola, {nome}! Bem-vindo a API."}