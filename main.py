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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    frage: str

# Globale Variablen
rag_chain = None
system_status = "Startet..."

# --- DIE NEUE HINTERGRUND-FUNKTION ---
def lade_ki_im_hintergrund():
    global rag_chain, system_status
    try:
        all_docs = []
        for file in os.listdir("."):
            if file.endswith(".pdf"):
                loader = PyPDFLoader(file)
                all_docs.extend(loader.load())

        if not all_docs:
            system_status = "Fehler: Keine PDFs gefunden!"
            return

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(all_docs)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=os.getenv("GOOGLE_API_KEY"))

        template = """Du bist der professionelle KI-Karriere-Assistent von Shahim Quraishy. 
        Antworte professionell, in der dritten Person und NUR auf Basis des Kontexts.
        Wenn eine Info fehlt, sag: "Dazu liegen mir keine Informationen vor."
        
        KONTEXT:
        {context}
        
        FRAGE: {question}
        
        ANTWORT:"""
        
        prompt = ChatPromptTemplate.from_template(template)

        rag_chain = (
            {"context": vectorstore.as_retriever(search_kwargs={"k": 6}) | (lambda docs: "\n\n".join(d.page_content for d in docs)), 
             "question": RunnablePassthrough()}
            | prompt | llm | StrOutputParser()
        )
        system_status = "Bereit"
        print("✅ KI IST JETZT EINSATZBEREIT!")
    except Exception as e:
        system_status = f"Fehler: {str(e)}"
        print(system_status)

# --- SERVER START ---
@app.on_event("startup")
async def startup_event():
    # Wir starten das Laden im Hintergrund. 
    # Dadurch blockiert der Server nicht und Render meldet sofort "Live 🎉"
    threading.Thread(target=lade_ki_im_hintergrund).start()

@app.get("/")
def home():
    return {"status": system_status}

@app.post("/chat")
async def chat(request: ChatRequest):
    # Falls jemand sofort fragt, während die PDFs noch im Hintergrund laden:
    if rag_chain is None: 
        return {"antwort": "Ich sortiere gerade noch Shahims Unterlagen. Bitte stell deine Frage in 30 Sekunden nochmal! ⏳"}
    
    return {"antwort": rag_chain.invoke(request.frage)}