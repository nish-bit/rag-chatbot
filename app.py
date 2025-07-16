import os
import warnings
import logging
import streamlit as st
import base64

from langchain_groq import ChatGroq
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator

# ---------------- GOOGLE LOGIN AUTH -------------------

user = st.experimental_user  # Only works on Streamlit Cloud
ALLOWED_USERS = ["nishant@example.com", "user1@example.com", "test@example.com"]
ADMIN_USERS = ["nishant@example.com"]

if user is None:
    st.warning("🔐 Please log in with your Google account to use this chatbot.")
    st.stop()

if user.email.lower() not in [email.lower() for email in ALLOWED_USERS]:
    st.error(f"🚫 Access denied for {user.email}")
    st.stop()

# ---------------- UI HEADER -------------------

st.title("🤖 Secure RAG Chatbot")
st.success(f"✅ Logged in as: {user.email}")
if st.button("🚪 Logout"):
    st.session_state.clear()
    st.experimental_rerun()

# ---------------- MODEL SELECTION -------------------

model_choice = st.sidebar.radio(
    "Choose your LLM:",
    ("Groq - LLaMA 3", "OpenAI - GPT-3.5"),
    index=0
)

# ---------------- ADMIN PANEL -------------------

if user.email.lower() in [email.lower() for email in ADMIN_USERS]:
    st.sidebar.subheader("👑 Admin Panel")
    if "user_messages" in st.session_state:
        selected_user = st.sidebar.selectbox("Select user to view chat:", list(st.session_state.user_messages.keys()))
        st.sidebar.write("Chat history:")
        for msg in st.session_state.user_messages[selected_user]:
            icon = "🧑" if msg['role'] == 'user' else "🤖"
            st.sidebar.markdown(f"{icon} **{msg['role'].capitalize()}**: {msg['content']}")
    else:
        st.sidebar.write("No messages yet.")

# ---------------- CHAT MEMORY -------------------

if "user_messages" not in st.session_state:
    st.session_state.user_messages = {}
if user.email not in st.session_state.user_messages:
    st.session_state.user_messages[user.email] = []

for msg in st.session_state.user_messages[user.email]:
    st.chat_message(msg['role']).markdown(msg['content'])

# ---------------- PDF UPLOAD -------------------

uploaded_files = st.file_uploader("📄 Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)

@st.cache_resource(show_spinner="🔄 Indexing PDFs...")
def create_vectorstore_from_pdfs(files):
    try:
        loaders = []
        for file in files:
            path = f"temp_{file.name}"
            with open(path, "wb") as f:
                f.write(file.read())
            loaders.append(PyPDFLoader(path))

        index = VectorstoreIndexCreator(
            embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2"),
            text_splitter=RecursiveCharacterTextSplitter(chunk_size=100000, chunk_overlap=100)
        ).from_loaders(loaders)
        return index.vectorstore
    except Exception as e:
        st.error(f"Vectorstore creation error: {e}")
        return None

# ---------------- CHAT INPUT -------------------

prompt = st.chat_input("Ask your question from the uploaded PDFs")

if prompt:
    if not uploaded_files:
        st.warning("📎 Please upload at least one PDF to proceed.")
        st.stop()

    st.chat_message("user").markdown(prompt)
    st.session_state.user_messages[user.email].append({"role": "user", "content": prompt})

    try:
        vectorstore = create_vectorstore_from_pdfs(uploaded_files)
        if vectorstore is None:
            raise Exception("Vectorstore is empty")

        if model_choice == "Groq - LLaMA 3":
            groq_api_key = os.getenv("GROQ_API_KEY") or "your_groq_key"
            chat_model = ChatGroq(api_key=groq_api_key, model_name="llama3-8b-8192")
        else:
            openai_api_key = os.getenv("OPENAI_API_KEY") or "your_openai_key"
            chat_model = ChatOpenAI(api_key=openai_api_key, model_name="gpt-3.5-turbo")

        chain = RetrievalQA.from_chain_type(
            llm=chat_model,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )

        result = chain({"query": prompt})
        response = result["result"]

        st.chat_message("assistant").markdown(response)
        st.session_state.user_messages[user.email].append({"role": "assistant", "content": response})

        # ---------------- SAVE CHAT -------------------
        chat_lines = [
            f"{m['role'].capitalize()}: {m['content']}"
            for m in st.session_state.user_messages[user.email]
        ]
        history_text = "\n\n".join(chat_lines)

        filename = f"chat_history_{user.email.replace('@', '_at_')}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(history_text)

        # ---------------- DOWNLOAD BUTTON -------------------
        b64 = base64.b64encode(history_text.encode()).decode()
        href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">📥 Download Your Chat History</a>'
        st.markdown(href, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")



