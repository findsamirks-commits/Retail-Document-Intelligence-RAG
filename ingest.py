import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

# Initialize environment
load_dotenv()

# 1. Load the Document
print("📄 Loading PDF document...")
loader = PyPDFLoader("document.pdf")
docs = loader.load()
print(f"Read {len(docs)} pages.")

# 2. Chunk the Text
# Overlap ensures context isn't lost if a sentence is cut in half
print("✂️ Splitting text into overlapping chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# 3. Generate Embeddings and Save to ChromaDB
print("🧠 Generating embeddings and building Vector DB...")
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# This creates a local folder named "chroma_db" to store the vectors
vectorstore = Chroma.from_documents(
    documents=splits, 
    embedding=embeddings, 
    persist_directory="./chroma_db"
)

print(f"✅ Successfully embedded {len(splits)} chunks into the local Chroma database!")