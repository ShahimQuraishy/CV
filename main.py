import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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
is_loading = False

def init_ai():
    global rag_chain, is_loading
    if rag_chain is not None or is_loading:
        return
    
    is_loading = True
    print("Starte KI-Initialisierung...")
    
    try:
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.runnables import RunnablePassthrough
        from langchain_core.output_parsers import StrOutputParser

        all_docs = []
        for file in os.listdir("."):
            if file.endswith(".pdf"):
                loader = PyPDFLoader(file)
                all_docs.extend(loader.load())

        if not all_docs:
            print("Keine PDFs gefunden!")
            is_loading = False
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
        print("✅ KI IST EINSATZBEREIT!")
    except Exception as e:
        print(f"Fehler bei KI-Init: {e}")
    finally:
        is_loading = False

@app.get("/")
def home():
    return {"status": "Server ist online. KI lädt bei der ersten Anfrage."}

@app.post("/chat")
async def chat(request: ChatRequest):
    global rag_chain, is_loading
    
    # Beim allerersten Aufruf starten wir die KI (Lazy Loading)
    if rag_chain is None:
        if not is_loading:
            import threading
            threading.Thread(target=init_ai).start()
        
        # Senden einer Zwischennachricht, während die KI hochfährt
        return {"antwort": "Ich aktiviere gerade mein Wissen (das dauert beim ersten Mal ca. 30 Sekunden). Bitte stelle deine Frage gleich nochmal! ⏳"}
    
    # Wenn die KI bereit ist
    try:
        antwort = rag_chain.invoke(request.frage)
        return {"antwort": antwort}
    except Exception as e:
        return {"antwort": f"Es gab ein Problem bei der Anfrage: {e}"}