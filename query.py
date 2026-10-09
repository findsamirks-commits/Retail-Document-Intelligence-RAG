import os
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

# 1. Load the persistent vector store using the same embedding model
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# 2. Configure retriever to fetch top 3 most relevant chunks
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 3. Initialize Gemini 2.5 Flash
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)

def ask_document(question: str):
    print(f"\n🔍 Searching vector store for: '{question}'...")
    relevant_docs = retriever.invoke(question)
    
    # Compile retrieved chunks with page metadata
    context_blocks = []
    for doc in relevant_docs:
        page_num = doc.metadata.get("page", 0) + 1  # 0-indexed to 1-indexed
        context_blocks.append(f"[Page {page_num}]:\n{doc.page_content}")
    
    combined_context = "\n\n---\n\n".join(context_blocks)
    
    # Strictly ground the LLM on provided context
    prompt = f"""
    You are an expert internal assistant. Answer the user's question accurately using ONLY the context provided below.
    If the context does not contain the answer, state clearly that the document does not mention it.
    Always cite the relevant page number(s) at the end of your answer.

    Context:
    {combined_context}

    Question: {question}

    Answer:
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content, relevant_docs

if __name__ == "__main__":
    # Test specific retail analytics queries
    test_question = "Which spice category generated the highest sales in Vasant Kunj, and what was its sales figure and mix percentage?"
    answer, docs = ask_document(test_question)
    
    print("\n💡 Response:")
    print(answer)