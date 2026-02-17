import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from pinecone import Pinecone

# --- UI Setup ---
st.set_page_config(page_title="AI FAQ Chatbot", layout="centered")
st.title("📚 Knowledge-Based FAQ Bot")

# --- Secrets (For Streamlit Cloud) ---
# When running locally, replace these or use a .env file
GROQ_API_KEY = st.sidebar.text_input("Groq API Key", type="password")
PINECONE_API_KEY = st.sidebar.text_input("Pinecone API Key", type="password")
INDEX_NAME = "chatbot-index"

if not GROQ_API_KEY or not PINECONE_API_KEY:
    st.info("Please add your API keys in the sidebar to continue.", icon="🗝️")
    st.stop()

# --- Initialize Embeddings & LLM ---
# We use HuggingFace for FREE local embeddings (no OpenAI cost)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = ChatGroq(api_key=GROQ_API_KEY, model_name="llama3-8b-8192")

# --- File Upload Section ---
uploaded_file = st.file_uploader("Upload Knowledge (PDF or DOCX)", type=["pdf", "docx"])

if uploaded_file is not None:
    with st.spinner("Processing document..."):
        # Save file to a temp location
        temp_path = "temp_doc"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Load data
        if uploaded_file.name.lower().endswith(".pdf"):
            loader = PyPDFLoader(temp_path)
        elif uploaded_file.name.lower().endswith(".docx"):
            loader = Docx2txtLoader(temp_path)
        else:
            st.error("Unsupported file type.")
            st.stop()
        
        data = loader.load()

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(data)

        # Upload to Pinecone
        try:
            vectorstore = PineconeVectorStore.from_documents(
                docs, 
                embeddings, 
                index_name=chatbot-index, 
                pinecone_api_key=pcsk_4Xgh95_UVfQ8UcYVnVArDhmgkyQgBzTmPdsGMma46anhN3gsKGTQBCr2EwXoSobtQrZv49N
            )
            st.success("Knowledge Base Updated!")
        except Exception as e:
            st.error(f"Error uploading documents to Pinecone: {e}")

# --- Chat Section ---
query = st.text_input("Ask a question about your documents:")

if query:
    try:
        # Connect to existing index
        pc = Pinecone(api_key=pcsk_4Xgh95_UVfQ8UcYVnVArDhmgkyQgBzTmPdsGMma46anhN3gsKGTQBCr2EwXoSobtQrZv49N)
        vectorstore = PineconeVectorStore(
            index_name=chatbot-index, 
            embedding=embeddings, 
            pinecone_api_key=pcsk_4Xgh95_UVfQ8UcYVnVArDhmgkyQgBzTmPdsGMma46anhN3gsKGTQBCr2EwXoSobtQrZv49N
        )
        
        # Setup Retrieval Chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
        )
        
        with st.spinner("Thinking..."):
            response = qa_chain.invoke(query)
            st.markdown("### Answer:")
            st.write(response["result"] if "result" in response else response)
    except Exception as e:
        st.error(f"Error retrieving answer: {e}")