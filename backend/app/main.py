from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

app = FastAPI(title="Textbook RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CHROMA_PATH = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

def format_docs(docs):
    formatted = []
    for doc in docs:
        page = doc.metadata.get("page", "?")
        chapter = doc.metadata.get("chapter", "?")
        formatted.append(f"[Page {page}, {chapter}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)

class ChatRequest(BaseModel):
    query: str

class NotesRequest(BaseModel):
    topic: str

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Textbook RAG API is running"}

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    try:
        vectorstore = get_vectorstore()
        llm = get_llm()
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        system_prompt = (
            "You are a helpful and highly knowledgeable study assistant analyzing an 800-page textbook.\n"
            "Use the provided context to answer the user's question accurately and comprehensively.\n"
            "If the answer is not contained in the context, say 'I cannot find the answer in the textbook text.'\n"
            "Crucially, YOU MUST INCLUDE CITATIONS for your claims. The context blocks provide the page number and chapter title.\n"
            "Format your citations inline like: [Page 42, Chapter 3: Introduction].\n\n"
            "Context:\n{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}"),
        ])

        # Retrieve documents
        docs = retriever.invoke(req.query)
        context = format_docs(docs)

        # Build chain using LCEL
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": req.query})
        
        sources = []
        for doc in docs:
            sources.append({
                "page": doc.metadata.get("page", "Unknown"),
                "chapter": doc.metadata.get("chapter", "Unknown"),
                "snippet": doc.page_content[:200] + "..."
            })
            
        return {
            "answer": answer,
            "sources": sources
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/notes")
async def notes_endpoint(req: NotesRequest):
    try:
        vectorstore = get_vectorstore()
        llm = get_llm()
        retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
        
        system_prompt = (
            "You are an expert tutor. Generate detailed, structured study notes on the following topic based ONLY on the provided context.\n"
            "Include bullet points, key definitions, and important concepts.\n"
            "Include citations for each main section: [Page X, Chapter Y].\n\n"
            "Context:\n{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Generate study notes for: {topic}")
        ])
        
        docs = retriever.invoke(req.topic)
        context = format_docs(docs)
        
        chain = prompt | llm | StrOutputParser()
        notes = chain.invoke({"context": context, "topic": req.topic})
        
        return {
            "notes": notes
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
