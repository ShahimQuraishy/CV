import os
import threading
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

app = FastAPI()

# CORS – für Produktion auf deine Domain einschränken
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # später gegen z.B. "https://shahimquraishy.github.io" austauschen
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    frage: str

# Globale Variablen
rag_chain = None
system_status = "Starte KI im Hintergrund..."

def lade_ki_im_hintergrund():
    global rag_chain, system_status
    try:
        all_docs = []
        # Akzeptiere sowohl .pdf als auch .de Dateien
        for file in os.listdir("."):
            if file.endswith((".pdf", ".de")):
                print(f"Lade Dokument: {file}")
                loader = PyPDFLoader(file)
                all_docs.extend(loader.load())

        if not all_docs:
            system_status = "Fehler: Keine PDF- oder .de-Dateien gefunden!"
            print(system_status)
            return

        # Text splitten
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(all_docs)

        # Embeddings & Vektorstore
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)

        # LLM mit API-Key aus Umgebungsvariable
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            system_status = "Fehler: GOOGLE_API_KEY nicht gesetzt!"
            print(system_status)
            return

        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0,
            google_api_key=api_key
        )

        # Prompt – streng kontextbasiert
        template = """Du bist der professionelle KI-Karriere-Assistent von Shahim Quraishy. 
        Antworte professionell, in der dritten Person und NUR auf Basis des Kontexts.
        Wenn eine Info fehlt, sag: "Dazu liegen mir keine Informationen vor."

        KONTEXT:
        {context}

        FRAGE: {question}

        ANTWORT:"""

        prompt = ChatPromptTemplate.from_template(template)

        # Retriever mit Formatierung
        retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
        def format_docs(docs):
            return "\n\n".join(d.page_content for d in docs)

        # Chain bauen
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        system_status = "Bereit"
        print("✅ KI IST JETZT EINSATZBEREIT!")

    except Exception as e:
        system_status = f"Fehler beim Laden der KI: {str(e)}"
        print(system_status)

# Start im Hintergrund – damit der Server sofort antwortet
@app.on_event("startup")
async def startup_event():
    threading.Thread(target=lade_ki_im_hintergrund).start()

@app.get("/")
def home():
    return {"status": system_status}

@app.post("/chat")
async def chat(request: ChatRequest):
    if rag_chain is None:
        return {"antwort": "Ich aktiviere gerade mein Wissen (das dauert beim ersten Mal ca. 30 Sekunden). Bitte stelle deine Frage gleich nochmal! ⏳"}
    try:
        antwort = rag_chain.invoke(request.frage)
        return {"antwort": antwort}
    except Exception as e:
        return {"antwort": f"Fehler bei der Anfrage: {str(e)}"}