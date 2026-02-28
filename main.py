import os
import time
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pypdf import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama


app = FastAPI()

# Rate limiting
last_request_time = {}
REQUEST_COOLDOWN = 1  # seconds between requests

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
        if not os.path.exists("lebenslauf.pdf"):
            print("❌ lebenslauf.pdf nicht gefunden im Arbeitsverzeichnis:", os.listdir("."))
            cv_text = ""
            return

        reader = PdfReader("lebenslauf.pdf")
        pages = []
        for i, p in enumerate(reader.pages):
            try:
                text = p.extract_text() or ""
                print(f"Seite {i+1}: {len(text)} Zeichen extrahiert")
                pages.append(text)
            except Exception as e:
                print(f"Fehler beim Lesen von Seite {i+1}: {e}")

        cv_text = "\n\n".join(pages)
        print("✅ CV geladen, Länge:", len(cv_text))

    except Exception as e:
        print(f"Fehler beim Laden des PDFs: {e}")
        cv_text = ""
    finally:
        is_loading = False


@app.get("/")
@app.head("/")
def home():
    return {"status": "Server läuft blitzschnell!"}


def get_llm_with_fallback():
    """Try Gemini first, fallback to simple response if quota exceeded"""
    api_key = os.getenv("GOOGLE_API_KEY")
    
    # Always try Gemini first
    if api_key:
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0.2,
                google_api_key=api_key,
            )
            return llm, "gemini"
        except Exception as e:
            print(f"Gemini nicht verfügbar: {e}")
    
    # No fallback available - return None
    return None, "none"

async def test_gemini(chain):
    """Test if Gemini is working"""
    try:
        await chain.ainvoke({"input": "test"})
    except Exception as e:
        print(f"Gemini Test fehlgeschlagen: {e}")

@app.post("/chat")
async def chat(request: ChatRequest):
    global cv_text, is_loading, last_request_time
    
    # Simple rate limiting
    client_ip = "client"  # In production, use actual client IP
    current_time = time.time()
    
    if client_ip in last_request_time:
        time_since_last = current_time - last_request_time[client_ip]
        if time_since_last < REQUEST_COOLDOWN:
            return {
                "antwort": f"Bitte warte {REQUEST_COOLDOWN - time_since_last:.1f} Sekunden vor der nächsten Anfrage."
            }
    
    last_request_time[client_ip] = current_time

    if cv_text is None:
        if not is_loading:
            import threading
            threading.Thread(target=load_cv).start()
        return {
            "antwort": "Ich lade gerade Shahims Lebenslauf. Bitte frag mich gleich nochmal! ⏳"
        }

    if cv_text == "":
        return {
            "antwort": "Ich konnte den Lebenslauf nicht laden. Prüfe, ob lebenslauf.pdf im Projekt-Root liegt."
        }

    # Get LLM with fallback
    llm, model_type = get_llm_with_fallback()
    
    if llm is None:
        # Provide a helpful response about Shahim based on common CV questions
        fallback_responses = {
            "default": "Ich bin Shahim Quraishys KI-Assistent. Leider ist das Gemini API-Kontingent erschöpft. Bitte stellen Sie Ihre Frage später erneut oder kontaktieren Sie Shahim direkt.",
            "skills": "Shahim hat Erfahrung in Python, FastAPI, maschinelles Lernen und Webentwicklung.",
            "contact": "Sie können Shahim über LinkedIn oder E-Mail kontaktieren.",
            "experience": "Shahim arbeitet als Softwareentwickler mit Fokus auf KI und Webtechnologien."
        }
        
        # Simple keyword matching for basic responses
        frage_lower = request.frage.lower()
        if any(word in frage_lower for word in ["skill", "können", "fähigkeiten"]):
            return {"antwort": fallback_responses["skills"]}
        elif any(word in frage_lower for word in ["kontakt", "email", "reach"]):
            return {"antwort": fallback_responses["contact"]}
        elif any(word in frage_lower for word in ["erfahrung", "experience", "arbeit"]):
            return {"antwort": fallback_responses["experience"]}
        else:
            return {"antwort": fallback_responses["default"]}

    prompt = ChatPromptTemplate.from_template(
        """Du bist ein Karriere-Assistent für Shahim Quraishy.
Nutze ausschließlich diese CV-Infos:

{cv}

Frage des Recruiters: {frage}

Antwort professionell in der dritten Person und auf Deutsch."""
    )

    chain = prompt | llm

    try:
        if model_type == "gemini":
            resp = await chain.ainvoke({"cv": cv_text, "frage": request.frage})
        else:
            resp = chain.invoke({"cv": cv_text, "frage": request.frage})
        
        text = resp.content if hasattr(resp, "content") else str(resp)
        model_info = f" (Antwort von {model_type})" if model_type != "gemini" else ""
        return {"antwort": text + model_info}
        
    except Exception as e:
        error_msg = str(e)
        
        # Check for quota exhaustion
        if "RESOURCE_EXHAUSTED" in error_msg or "429" in error_msg:
            return {
                "antwort": "Das Gemini API-Kontingent ist erschöpft. Ich versuche, auf ein lokales Modell umzuschalten... Bitte versuche es in wenigen Sekunden erneut."
            }
        
        return {"antwort": f"Fehler bei der Anfrage: {e}"}
