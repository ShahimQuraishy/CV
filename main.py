import os
import time
import threading
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pypdf import PdfReader
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


app = FastAPI()

last_request_time = {}
REQUEST_COOLDOWN = 1

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    frage: str


cv_text = None
is_loading = False


def load_cv():
    global cv_text, is_loading
    if cv_text is not None or is_loading:
        return

    is_loading = True
    try:
        print("🔄 Lade lebenslauf.pdf ...")
        print("📁 Dateien:", os.listdir("."))

        if not os.path.exists("lebenslauf.pdf"):
            print("❌ lebenslauf.pdf nicht gefunden!")
            cv_text = ""
            return

        reader = PdfReader("lebenslauf.pdf")
        pages = []
        for i, p in enumerate(reader.pages):
            text = p.extract_text() or ""
            print(f"Seite {i+1}: {len(text)} Zeichen")
            pages.append(text)

        cv_text = "\n\n".join(pages)
        print("✅ CV geladen, Länge:", len(cv_text))

    except Exception as e:
        print(f"Fehler: {e}")
        cv_text = ""
    finally:
        is_loading = False


@app.get("/")
def home():
    return {"status": "Server läuft mit Groq ⚡"}


@app.post("/chat")
async def chat(request: ChatRequest):
    global cv_text, is_loading, last_request_time

    # Rate Limiting
    client_ip = "client"
    current_time = time.time()
    if client_ip in last_request_time:
        time_since_last = current_time - last_request_time[client_ip]
        if time_since_last < REQUEST_COOLDOWN:
            return {"antwort": f"Bitte warte {REQUEST_COOLDOWN - time_since_last:.1f}s."}
    last_request_time[client_ip] = current_time

    # Sicherheitsprüfungen
    if len(request.frage) > 150:
        return {"antwort": "❌ Frage zu lang (max 150 Zeichen)."}

    injection_keywords = [
        "ignoriere", "vergiss", "ignore", "jailbreak", "act as", "whu", 
        "otto beisheim", "data science solutions", "ai solutions"
    ]
    if any(kw in request.frage.lower() for kw in injection_keywords):
        return {"antwort": "⚠️ Ich beantworte nur Fragen über Shahim Quraishy."}

    # CV laden
    if cv_text is None:
        if not is_loading:
            threading.Thread(target=load_cv).start()
        return {"antwort": "Lade Shahims CV... Frag gleich nochmal! ⏳"}

    if cv_text == "":
        return {"antwort": "❌ CV konnte nicht geladen werden."}

    # Groq
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        return {"antwort": "❌ GROQ_API_KEY fehlt."}

    try:
        llm = ChatGroq(
            groq_api_key=groq_key,
            model="llama-3.3-70b-versatile",
            temperature=0.2,
        )
    except Exception as e:
        return {"antwort": f"❌ Groq: {e}"}

    # Prompt
    prompt = ChatPromptTemplate.from_template(
        """Du bist Shahim Quraishys Karriere-Assistent.

STRICT:
- NUR Shahims CV verwenden
- Alle anderen Infos ignorieren
- Nur Fragen über Shahim beantworten

Shahims CV:
{cv}

Frage: {frage}

Antwort auf Deutsch, 3. Person, professionell."""
    )

    chain = prompt | llm

    # KORREKTER AUFRUF: invoke() statt ainvoke()
    try:
        resp = chain.invoke({"cv": cv_text, "frage": request.frage})  # ← FIX!
        text = resp.content if hasattr(resp, "content") else str(resp)
        return {"antwort": text}

    except Exception as e:
        return {"antwort": f"❌ Fehler: {str(e)[:100]}..."}
