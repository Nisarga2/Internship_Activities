from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import requests

# ---------------- APP ----------------
app = FastAPI(title="Ollama Chat (100% Free & Local)")
templates = Jinja2Templates(directory="templates")

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3"

# ---------------- REQUEST MODEL ----------------
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

# ---------------- MEMORY ----------------
memory = {}

# ---------------- UI ----------------
@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

# ---------------- CHAT ----------------
@app.post("/chat")
def chat(req: ChatRequest):
    if req.session_id not in memory:
        memory[req.session_id] = []

    memory[req.session_id].append(f"User: {req.message}")

    prompt = "\n".join(memory[req.session_id]) + "\nAssistant:"

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload)

    if response.status_code != 200:
        return {"reply": "Ollama is not running. Start it with `ollama run llama3`"}

    data = response.json()
    reply = data.get("response", "").strip()

    memory[req.session_id].append(f"Assistant: {reply}")

    return {"reply": reply}
