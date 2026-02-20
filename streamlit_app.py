import os
import time
import streamlit as st

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

from pinecone import Pinecone, ServerlessSpec

st.set_page_config(page_title="AI FAQ Chatbot", layout="centered")
st.title("📚 Knowledge‑Based FAQ Bot")

st.sidebar.header("🔐 API Keys")
st.sidebar.caption("Use Streamlit ‘Secrets’ in production. Sidebar is for local testing only.")

# Prefer secrets in production
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.sidebar.text_input("Groq API Key", type="password")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or st.sidebar.text_input("Pinecone API Key", type="password")

INDEX_NAME = "chatbot-index"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"

if not GROQ_API_KEY or not PINECONE_API_KEY:
    st.info("Add your API keys in the sidebar (local) or via **Secrets** in Streamlit Cloud.", icon="🗝️")
    st.stop()

# Embeddings: MiniLM-L6-v2 (384-dim)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Groq LLM (fast model)
llm = ChatGroq(api_key=GROQ_API_KEY, model="llama3-8b-8192", temperature=0)

# Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Ensure index exists
if not pc.has_index(INDEX_NAME):
    with st.spinner("Creating Pinecone index (one‑time)…"):
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
        # Wait for readiness
        for _ in range(60):
            time.sleep(1)
            if pc.describe_index(INDEX_NAME).status.get("ready"):
                break

index = pc.Index(INDEX_NAME)
vectorstore = PineconeVectorStore(index=index, embedding=embeddings)

uploaded_files = st.file_uploader("Upload Knowledge (PDF or DOCX)", type=["pdf", "docx"], accept_multiple_files=True)

if uploaded_files:
    with st.spinner("Processing & indexing document(s)…"):
        docs_all = []
        for file in uploaded_files:
            tmp_path = f"tmp_{file.name}"
            with open(tmp_path, "wb") as f:
                f.write(file.getbuffer())

            if file.name.lower().endswith(".pdf"):
                loader = PyPDFLoader(tmp_path)
            else:
                loader = Docx2txtLoader(tmp_path)

            docs = loader.load()
            docs_all.extend(docs)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=120)
        chunks = text_splitter.split_documents(docs_all)

        vectorstore.add_documents(chunks)

    st.success("✅ Knowledge Base Updated!")

query = st.text_input("Ask a question about your documents:")

if query:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )

    with st.spinner("Thinking…"):
        response = qa_chain.invoke({"query": query})

    st.markdown("### Answer")
    st.write(response["result"])

    with st.expander("Sources"):
        for i, d in enumerate(response.get("source_documents", []), 1):
            st.markdown(f"**{i}.** {d.metadata.get('source','(no source)')}")
