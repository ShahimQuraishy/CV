import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


app = FastAPI()

# CORS, damit dein HTML von GitHub/Render auf /chat zugreifen darf
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    frage: str


rag_chain = None
is_loading = False


def init_ai():
    global rag_chain, is_loading
    if rag_chain is not None or is_loading:
        return

    is_loading = True
    print("Starte KI-Initialisierung mit leichtem Google-Modell...")

    try:
        all_docs = []

        # Alle PDFs im Projekt-Root laden (z.B. lebenslauf.pdf)
        for file in os.listdir("."):
            if file.lower().endswith(".pdf"):
                print(f"Lade: {file}")
                loader = PyPDFLoader(file)
                all_docs.extend(loader.load())

        if not all_docs:
            print("Keine PDFs gefunden!")
            is_loading = False
            return

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        splits = text_splitter.split_documents(all_docs)

        api_key = os.getenv("GOOGLE_API_KEY")

        # Embeddings + Vektorindex
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key,
        )
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)

        # Leichtes Gemini-Modell
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0,
            google_api_key=api_key,
        )

        template = """Du bist der professionelle KI-Karriere-Assistent von Shahim Quraishy.
Antworte professionell, in der dritten Person und NUR auf Basis des Kontexts.
Wenn eine Info fehlt, sag: "Dazu liegen mir keine Informationen vor."

KONTEXT:
{context}

FRAGE: {question}

ANTWORT:"""

        prompt = ChatPromptTemplate.from_template(template)

        rag_chain_local = (
            {
                "context": vectorstore.as_retriever(search_kwargs={"k": 6})
                | (lambda docs: "\n\n".join(d.page_content for d in docs)),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        rag_chain = rag_chain_local
        print("✅ KI IST EINSATZBEREIT!")

    except Exception as e:
        print(f"Fehler bei KI-Init: {e}")
    finally:
        is_loading = False


# Health-Check für Render (GET/HEAD auf "/")
@app.get("/")
@app.head("/")
def home():
    return {"status": "Server läuft blitzschnell!"}


@app.post("/chat")
async def chat(request: ChatRequest):
    global rag_chain, is_loading

    # Lazy-Load der KI beim ersten Request
    if rag_chain is None:
        if not is_loading:
            import threading

            threading.Thread(target=init_ai).start()

        return {
            "antwort": "Ich überfliege gerade Shahims Lebenslauf (dauert nur wenige Sekunden). Bitte frag mich das gleich nochmal! ⏳"
        }

    try:
        answer = rag_chain.invoke(request.frage)
        return {"antwort": answer}
    except Exception as e:
        return {"antwort": f"Fehler bei der Anfrage: {e}"}
