import os
import tempfile
import streamlit as st
import google.generativeai as genai

# Modern LangChain Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.embeddings import Embeddings

# =========================================================
# ⚙️ PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Enterprise RAG - Created by Saima",
    page_icon="🧠",
    layout="wide"
)

# Custom Styling
st.markdown("""
    <style>
    .main { background-color: #0f172a; color: #f8fafc; }
    .stChatInput { border-radius: 12px; }
    .source-box { 
        background-color: #1e293b; 
        border-left: 4px solid #38bdf8; 
        padding: 10px; 
        margin-top: 5px;
        border-radius: 4px;
        font-size: 0.85em;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 Enterprise RAG Intelligence System")
st.caption("⚡ Upload any PDF / Research Paper & Chat with 100% Citation Accuracy")

# Custom Direct Gemini Embeddings (Bypasses LangChain Bug & PyTorch requirement)
class DirectGeminiEmbeddings(Embeddings):
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for text in texts:
            res = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_document"
            )
            embeddings.append(res['embedding'])
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        res = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_query"
        )
        return res['embedding']

# =========================================================
# 🔑 SIDEBAR - API KEY, FILE UPLOAD & SAIMA'S BRANDING
# =========================================================
with st.sidebar:
    st.header("🔑 Authentication & Setup")

    api_key = st.text_input("Enter Google Gemini API Key:", type="password")
    st.markdown("[👉 Get Free Gemini API Key Here](https://aistudio.google.com/)")

    st.divider()

    st.header("📂 Document Ingestion")
    uploaded_files = st.file_uploader(
        "Upload PDF Documents",
        type=["pdf"],
        accept_multiple_files=True
    )

    # =========================================================
    # 👩‍💻 SAIMA'S DEVELOPER BRANDING
    # =========================================================
    st.divider()
    st.markdown("### 👩‍💻 Project Owner & Developer")
    st.markdown("**Developed by:** Saima")
    st.markdown("**Role:** AI & Python Developer")
    st.markdown("**Tech Stack:** LangChain | ChromaDB | Gemini 1.5 | Streamlit")
    st.markdown("📧 **Contact:** saima.developer@gmail.com")

# Session State Initializations
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None


# =========================================================
# 🔄 DOCUMENT PROCESSING & VECTOR DB INGESTION
# =========================================================
def process_documents(files, google_api_key):
    documents = []

    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file.read())
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        for doc in docs:
            doc.metadata["source_file"] = file.name

        documents.extend(docs)
        os.remove(tmp_path)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)

    # Direct Gemini API Embeddings
    embeddings = DirectGeminiEmbeddings(api_key=google_api_key)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="enterprise_rag"
    )

    return vectorstore


# Process Button Logic
if uploaded_files and api_key:
    if st.sidebar.button("🚀 Process & Index Documents"):
        with st.spinner("📄 Extracting Text, Generating Vector Embeddings..."):
            try:
                st.session_state.vectorstore = process_documents(uploaded_files, api_key)
                st.sidebar.success(f"✅ Success! Indexed {len(uploaded_files)} PDF(s).")
            except Exception as e:
                st.sidebar.error(f"Error processing files: {str(e)}")

# =========================================================
# 💬 RAG RETRIEVAL & CHAT INTERFACE
# =========================================================

# Display Chat History
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📌 Source Citations & References"):
                for src in message["sources"]:
                    html_code = f"<div class='source-box'><b>📄 File:</b> {src['file']} | <b>📖 Page:</b> {src['page']}<br><i>\"{src['content']}...\"</i></div>"
                    st.markdown(html_code, unsafe_allow_html=True)

# User Query Input
if query := st.chat_input("Ask anything from your documents..."):

    if not api_key:
        st.warning("⚠️ Please provide your Gemini API Key in the sidebar.")
        st.stop()

    if not st.session_state.vectorstore:
        st.warning("⚠️ Please upload and process at least one PDF first.")
        st.stop()

    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("🧠 Searching Vector Database & Reasoning..."):

            retriever = st.session_state.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            )
            retrieved_docs = retriever.invoke(query)

            context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

            prompt = ChatPromptTemplate.from_template("""You are an Enterprise AI Research Assistant created by Saima.
Use the following retrieved context pieces to answer the user's question.
If the answer is NOT present in the provided context, state clearly:
'I cannot find the answer in the uploaded documents.' Do not invent information.

Context:
{context}

Question: {input}
""")

            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=api_key,
                temperature=0.2
            )

            rag_chain = prompt | llm | StrOutputParser()
            answer = rag_chain.invoke({"context": context_text, "input": query})

            st.markdown(answer)

            sources = []
            for doc in retrieved_docs:
                sources.append({
                    "file": doc.metadata.get("source_file", "Unknown PDF"),
                    "page": doc.metadata.get("page", 0) + 1,
                    "content": doc.page_content[:200].replace("\n", " ")
                })

            if sources:
                with st.expander("📌 Source Citations & References"):
                    for src in sources:
                        html_code = f"<div class='source-box'><b>📄 File:</b> {src['file']} | <b>📖 Page:</b> {src['page']}<br><i>\"{src['content']}...\"</i></div>"
                        st.markdown(html_code, unsafe_allow_html=True)

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })
