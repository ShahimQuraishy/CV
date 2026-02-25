import os
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

# --- DAS HIER IST DIE WICHTIGE ZEILE ---
app = FastAPI()

# CORS Middleware (damit deine Website zugreifen darf)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    frage: str

# Globale Variable für unsere KI-Pipeline
rag_chain = None

@app.on_event("startup")
def startup_event():
    global rag_chain
    # Hier kommt dein restlicher Code (PDF laden, Embeddings, etc.)
    # ... (wie ich ihn dir oben geschickt habe)