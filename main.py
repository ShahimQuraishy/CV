import os
import time
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
        print("🔄 Versuche, lebenslauf.pdf zu laden ...")
        print("📁 Dateien im Verzeichnis:", os.listdir("."))

        if not os.path.exists("lebenslauf.pdf"):
            print("❌ lebenslauf.pdf nicht gefunden!")
            cv_text = ""
            return

        reader = PdfReader("lebenslauf.pdf")
        pages = []
        for i, p in enumerate(reader.pages):
            try:
                text = p.extract_text() or ""
                print(f"Seite {i+1}: {len(text)} Zeichen")
                pages.append(text)
            except Exception as e:
                print(f"Fehler Seite {i+1}: {e}")

        cv_text = "\n\n".join(pages)
        print("✅ CV geladen, Länge:", len(cv_text))

    except Exception as e:
        print(f"Fehler beim Laden: {e}")
        cv_text = ""
    finally:
        is_loading = False


@app.get("/")
@app.head("/")
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
            return {
                "antwort": f"Bitte warte {REQUEST_COOLDOWN - time_since_last:.1f} Sekunden."
            }
    last_request_time[client_ip] = current_time

    # CV laden
    if cv_text is None:
        if not is_loading:
            import threading
            threading.Thread(target=load_cv).start()
        return {"antwort": "Ich lade gerade Shahims Lebenslauf. Bitte frag mich gleich nochmal! ⏳"}

    if cv_text == "":
        return {"antwort": "❌ Lebenslauf konnte nicht geladen werden. Prüfe ob lebenslauf.pdf im Root liegt."}

    # Groq LLM
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        return {"antwort": "❌ GROQ_API_KEY fehlt. Bitte in Render Environment Variables setzen."}

    try:
        llm = ChatGroq(
            groq_api_key=groq_key,
            model="llama-3.1-70b-versatile",
            temperature=0.2,
        )
    except Exception as e:
        return {"antwort": f"❌ Groq konnte nicht initialisiert werden: {e}"}

    # Prompt
    prompt = ChatPromptTemplate.from_template(
        """Du bist ein Karriere-Assistent für Shahim Quraishy.
Nutze ausschließlich diese CV-Infos:

{cv}

Frage des Recruiters: {frage}

Antworte professionell in der dritten Person und auf Deutsch."""
    )

    chain = prompt | llm

    try:
        resp = await chain.ainvoke({"cv": cv_text, "frage": request.frage})
        text = resp.content if hasattr(resp, "content") else str(resp)
        return {"antwort": text}

    except Exception as e:
        error_msg = str(e)
        if "RESOURCE_EXHAUSTED" in error_msg or "429" in error_msg:
            return {"antwort": "⚠️ Groq Kontingent erschöpft. Bitte versuche es in 1 Minute erneut."}
        return {"antwort": f"❌ Fehler: {e}"}
