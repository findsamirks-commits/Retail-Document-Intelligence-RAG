import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Retail Document Intelligence", page_icon="📄", layout="wide")
st.title("📄 Retail Document Intelligence (RAG)")
st.caption("Grounded document Q&A powered by ChromaDB & Gemini 2.5 Flash")

# Initialize embeddings and ChromaDB retriever
@st.cache_resource
def load_retriever():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    return vectorstore.as_retriever(search_kwargs={"k": 4})

retriever = load_retriever()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)

# Maintain chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("Ask any question about your document (e.g., top selling spices, store comparison)...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant document chunks..."):
            relevant_docs = retriever.invoke(user_query)
            
            context_blocks = []
            citations = []
            for doc in relevant_docs:
                page_num = doc.metadata.get("page", 0) + 1
                citations.append(f"Page {page_num}")
                context_blocks.append(f"[Page {page_num}]:\n{doc.page_content}")
            
            combined_context = "\n\n---\n\n".join(context_blocks)
            
            prompt = f"""
            You are a retail analytics assistant. Answer the user question accurately using ONLY the context provided below.
            If the context does not contain enough information, state that clearly.
            Context:
            {combined_context}

            Question: {user_query}

            Answer:
            """
            
            response = llm.invoke([HumanMessage(content=prompt)])
            full_response = response.content + f"\n\n**Sources:** {', '.join(sorted(set(citations)))}"
            
            st.markdown(full_response)
            
            with st.expander("Inspected Context Chunks"):
                for doc in relevant_docs:
                    st.write(f"**Page {doc.metadata.get('page', 0) + 1}:**")
                    st.text(doc.page_content)

            st.session_state.messages.append({"role": "assistant", "content": full_response})