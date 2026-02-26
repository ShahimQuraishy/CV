import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# LangChain & Google imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

app = FastAPI()

# CORS: Erlaubt deiner Website den Zugriff
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    frage: str

# Globale Pipeline
rag_chain = None

@app.on_event("startup")
def startup_event():
    global rag_chain
    all_docs = []
    
    # 1. Dokumente laden (PDF und TXT)
    for file in os.listdir("."):
        if file.endswith(".pdf"):
            try:
                loader = PyPDFLoader(file)
                all_docs.extend(loader.load())
            except Exception as e:
                print(f"Fehler beim Laden von {file}: {e}")
        elif file.endswith(".txt"):
            try:
                loader = TextLoader(file, encoding="utf-8")
                all_docs.extend(loader.load())
            except Exception as e:
                print(f"Fehler beim Laden von {file}: {e}")

    if not all_docs:
        print("Keine Dokumente gefunden!")
        return

    # 2. Text-Splitting (Optimiert für Zusammenhänge)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(all_docs)

    # 3. Embeddings & Vectorstore (Lokal im RAM)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    
    # 4. LLM Konfiguration (Präzise auf Fakten getrimmt)
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", 
        temperature=0,  # 0 = Keine Fantasie, nur Fakten
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    # 5. Verbesserter System-Prompt für Recruiter
    template = """
    Du bist der exklusive KI-Karriere-Assistent von Shahim Quraishy. 
    Deine Aufgabe ist es, Fragen von Recruitern präzise und professionell auf Basis der bereitgestellten Dokumente zu beantworten.

    VERHALTENSREGELN:
    - Antworte immer auf DEUTSCH.
    - Antworte in der DRITTEN PERSON (z.B. "Shahim verfügt über...", "Er hat bei 1&1 gearbeitet...").
    - Nutze NUR die Informationen aus dem KONTEXT. Wenn etwas nicht drinsteht, sag: "Dazu liegen mir keine Informationen vor."
    - Sei präzise: Nenne konkrete Firmen (1&1, RTI, United Internet), Technologien und Zeiträume.
    - Bleib höflich und motiviert, wie ein erstklassiger Assistent.

    KONTEXT:
    {context}

    FRAGE:
    {question}

    ANTWORT:
    """
    
    prompt = ChatPromptTemplate.from_template(template)

    # 6. RAG-Chain Aufbau
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": vectorstore.as_retriever(search_kwargs={"k": 5}) | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    print("✅ RAG-Chain ist bereit!")

@app.get("/")
def home():
    return {"status": "Server läuft", "bot": "Shahim KI-Assistent"}

@app.post("/chat")
async def chat(request: ChatRequest):
    if rag_chain is None:
        return {"antwort": "System wird noch gestartet... Bitte in 10 Sekunden erneut versuchen."}
    
    try:
        # Hier wird die Antwort generiert
        ergebnis = rag_chain.invoke(request.frage)
        return {"antwort": ergebnis}
    except Exception as e:
        return {"antwort": f"Fehler bei der Anfrage: {str(e)}"}