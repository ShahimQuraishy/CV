import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pypdf import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

app = FastAPI()

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
        print("Lade lebenslauf.pdf ...")
        reader = PdfReader("lebenslauf.pdf")
        pages = [p.extract_text() or "" for p in reader.pages]
        cv_text = "\n\n".join(pages)
        print("✅ CV geladen!")
    except Exception as e:
        print(f"Fehler beim Laden des PDFs: {e}")
        cv_text = ""
    finally:
        is_loading = False

@app.get("/")
@app.head("/")
def home():
    return {"status": "Server läuft blitzschnell!"}

@app.post("/chat")
async def chat(request: ChatRequest):
    global cv_text, is_loading

    if cv_text is None:
        if not is_loading:
            import threading
            threading.Thread(target=load_cv).start()
        return {
            "antwort": "Ich lade gerade Shahims Lebenslauf. Bitte frag mich gleich nochmal! ⏳"
        }

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return {"antwort": "GOOGLE_API_KEY ist nicht gesetzt."}

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.2,
        google_api_key=api_key,
    )

    prompt = ChatPromptTemplate.from_template(
        """Du bist ein Karriere-Assistent für Shahim Quraishy.
Nutze ausschließlich diese CV-Infos:

{cv}

Frage des Recruiters: {frage}

Antwort professionell in der dritten Person und auf Deutsch."""
    )

    chain = prompt | llm

    try:
        resp = chain.invoke({"cv": cv_text, "frage": request.frage})
        text = resp.content if hasattr(resp, "content") else str(resp)
        return {"antwort": text}
    except Exception as e:
        return {"antwort": f"Fehler bei der Anfrage: {e}"}
