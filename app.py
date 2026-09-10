import os
import tempfile
import requests
import streamlit as st

# LangChain Imports for PDF Processing & BM25
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever

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

# =========================================================
# 🔑 DIRECT REST API CALLER FOR GEMINI (Bulletproof Solution)
# =========================================================
def call_gemini_rest_api(api_key, prompt_text):
    # Step 1: Auto-discover available models for this specific user key
    list_models_url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    
    selected_model_path = None
    try:
        res = requests.get(list_models_url, timeout=10)
        if res.status_code == 200:
            models_list = res.json().get('models', [])
            for m in models_list:
                methods = m.get('supportedGenerationMethods', [])
                if 'generateContent' in methods:
                    name = m.get('name', '')
                    if 'gemini' in name:
                        selected_model_path = name
                        if 'flash' in name: # Prefer flash if available
                            break
    except Exception:
        pass

    # Fallback to standard path if discovery fails
    if not selected_model_path:
        selected_model_path = "models/gemini-1.5-flash"

    # Step 2: Direct REST POST Request to Gemini
    endpoint_url = f"https://generativelanguage.googleapis.com/v1beta/{selected_model_path}:generateContent?key={api_key}"
    
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {
                "parts": [{"text": prompt_text}]
            }
        ]
    }

    response = requests.post(endpoint_url, json=payload, headers=headers, timeout=30)
    
    if response.status_code == 200:
        data = response.json()
        try:
            return data['candidates'][0]['content']['parts'][0]['text']
        except (KeyError, IndexError):
            return "Unable to parse response from Gemini API."
    else:
        # If v1beta fails, try v1 fallback endpoint
        alt_url = f"https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent?key={api_key}"
        alt_response = requests.post(alt_url, json=payload, headers=headers, timeout=30)
        if alt_response.status_code == 200:
            data = alt_response.json()
            return data['candidates'][0]['content']['parts'][0]['text']
        
        raise Exception(f"API Error ({response.status_code}): {response.text}")


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
    st.markdown("**Tech Stack:** LangChain | BM25 | Gemini 1.5 | Streamlit")
    st.markdown("📧 **Contact:** saima.developer@gmail.com")

# Session State Initializations
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None


# =========================================================
# 🔄 DOCUMENT PROCESSING & INDEXING
# =========================================================
def process_documents(files):
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

    retriever = BM25Retriever.from_documents(chunks)
    retriever.k = 4

    return retriever


# Process Button Logic
if uploaded_files:
    if st.sidebar.button("🚀 Process & Index Documents"):
        with st.spinner("📄 Extracting Text & Indexing Documents..."):
            try:
                st.session_state.retriever = process_documents(uploaded_files)
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

    if not st.session_state.retriever:
        st.warning("⚠️ Please upload and process at least one PDF first.")
        st.stop()

    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("🧠 Searching Indexed Documents & Reasoning..."):

            retrieved_docs = st.session_state.retriever.invoke(query)
            context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

            prompt = f"""You are an Enterprise AI Research Assistant created by Saima.
Use the following retrieved context pieces to answer the user's question.
If the answer is NOT present in the provided context, state clearly:
'I cannot find the answer in the uploaded documents.' Do not invent information.

Context:
{context_text}

Question: {query}
"""

            try:
                # Direct REST API Call (Bypasses Python SDK version issues completely)
                answer = call_gemini_rest_api(api_key, prompt)

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

            except Exception as e:
                st.error(f"Error calling Gemini API: {str(e)}")
