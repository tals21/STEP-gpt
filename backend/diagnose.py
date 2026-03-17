"""Diagnostic script to check retrieval quality."""
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

CHROMA_PATH = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

# Check what chapters exist in the metadata
collection = vectorstore._collection
all_metadata = collection.get(include=["metadatas"])
chapters = set()
for meta in all_metadata["metadatas"]:
    chapters.add(meta.get("chapter", "Unknown"))

print("=== CHAPTERS FOUND IN DATABASE ===")
for ch in sorted(chapters):
    print(f"  - {ch}")

print(f"\nTotal chunks: {collection.count()}")

# Test retrieval for different queries
test_queries = [
    "What is chapter 1 about?",
    "What are the key topics covered in the first chapter?",
    "Explain the mechanism of action of beta blockers",
    "What are gram positive bacteria?",
]

for query in test_queries:
    print(f"\n{'='*60}")
    print(f"QUERY: {query}")
    print(f"{'='*60}")
    results = vectorstore.similarity_search_with_score(query, k=3)
    for i, (doc, score) in enumerate(results):
        print(f"\n  Result {i+1} (score: {score:.4f}):")
        print(f"  Page: {doc.metadata.get('page')}, Chapter: {doc.metadata.get('chapter')}")
        print(f"  Content: {doc.page_content[:200]}...")
