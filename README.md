# 📄 Retail Document Intelligence (RAG)

An intelligent retrieval-augmented generation (RAG) application designed to strictly query internal retail documents and Standard Operating Procedures (SOPs). Built with LangChain, ChromaDB, and Gemini 2.5 Flash, this tool provides grounded, page-cited answers without hallucinating missing information.

## Key Features
* **Strict Grounding:** The LLM is constrained to answer *only* using the retrieved context from the uploaded PDF. It explicitly states if the document does not contain the answer.
* **Vector Retrieval:** Parses and chunks retail PDF reports (e.g., Regional Sales, Store Launch SOPs), generating vector embeddings via Google's `gemini-embedding-001` model.
* **Semantic Search:** Stores embeddings in a persistent, local ChromaDB database for fast semantic similarity retrieval.
* **Source Citations:** Automatically attaches page numbers and allows users to inspect the exact textual chunks retrieved from the document.

## Tech Stack
* **Interface**: Streamlit
* **Orchestration**: LangChain
* **Vector Database**: ChromaDB
* **LLM & Embeddings**: Google Gemini 2.5 Flash, `gemini-embedding-001`
