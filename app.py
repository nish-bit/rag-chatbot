import os
import warnings
import logging
import streamlit as st
import base64
from fpdf import FPDF
from datetime import datetime

from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator

# ---------------- GOOGLE LOGIN AUTH -------------------

user = getattr(st, "user", None)
if user is None or not hasattr(user, "email"):
    class MockUser:
        email = "test@example.com"
    user = MockUser()

ALLOWED_USERS = ["nishant@example.com", "user1@example.com", "test@example.com"]
ADMIN_USERS = ["nishant@example.com"]

if user.email.lower() not in [email.lower() for email in ALLOWED_USERS]:
    st.error(f"🚫 Access denied for {user.email}")
    st.stop()

# ---------------- UI HEADER -------------------

st.title("🤖 Secure RAG Chatbot")
st.success(f"✅ Logged in as: {user.email}")
if st.button("🚪 Logout"):
    st.session_state.clear()
    st.experimental_set_query_params(logout=1)
    st.stop()

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

if "feedback" not in st.session_state:
    st.session_state.feedback = {}

for idx, msg in enumerate(st.session_state.user_messages[user.email]):
    st.chat_message(msg['role']).markdown(msg['content'])
    if msg['role'] == 'assistant':
        col1, col2, col3 = st.columns([1, 1, 6])
        with col1:
            if st.button("👍", key=f"like_{idx}"):
                st.session_state.feedback[idx] = "like"
                st.toast("✅ Feedback saved: You liked the response.")
        with col2:
            if st.button("👎", key=f"dislike_{idx}"):
                st.session_state.feedback[idx] = "dislike"
                st.toast("⚠️ Feedback saved: You disliked the response.")
        with col3:
            st.text_input("💬 Comment", key=f"comment_{idx}")

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

        groq_api_key = "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"
        chat_model = ChatGroq(api_key=groq_api_key, model_name="llama3-8b-8192")

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

        # ---------------- SAVE CHAT TO PDF -------------------
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        chat_lines = [
            f"{m['role'].capitalize()}: {m['content']}"
            for m in st.session_state.user_messages[user.email]
        ]
        uploaded_names = ", ".join([file.name for file in uploaded_files]) if uploaded_files else "None"
        model_used = "Groq - llama3-8b-8192"

        pdf_filename = f"chat_history_{user.email.replace('@', '_at_')}.pdf"
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        pdf.cell(0, 10, f"Chat history for {user.email}", ln=True)
        pdf.cell(0, 10, f"Generated at: {now}", ln=True)
        pdf.cell(0, 10, f"Model Used: {model_used}", ln=True)
        pdf.cell(0, 10, f"PDFs Uploaded: {uploaded_names}", ln=True)
        pdf.ln()

        for line in chat_lines:
            for subline in line.split("\n"):
                pdf.multi_cell(0, 10, subline)
            pdf.ln()

        pdf.output(pdf_filename)

        with open(pdf_filename, "rb") as f:
            b64_pdf = base64.b64encode(f.read()).decode()
        href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="{pdf_filename}">📄 Download Chat History as PDF</a>'
        st.markdown(href, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")





