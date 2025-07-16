
import os
import warnings
import logging
import streamlit as st
import base64
from fpdf import FPDF
from datetime import datetime
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email.mime.text import MIMEText
import urllib.parse

from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.indexes import VectorstoreIndexCreator
from googletrans import Translator

# ---------------- PAGE CONFIG + STYLE -------------------

st.set_page_config(page_title="Secure RAG Chatbot", page_icon="🤖", layout="wide")
st.markdown("""
    <style>
        .main {
            background-color: #f4f4f4;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            padding: 8px 16px;
        }
        .stTextInput>div>input {
            border-radius: 10px;
        }
        .css-1d391kg, .stChatMessage, .css-1v0mbdj, .css-10trblm, .stTextArea, .stMarkdown {
            font-family: 'Segoe UI', sans-serif;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------- CHATBOT FUNCTIONALITY -------------------

st.title("🤖 Secure RAG Chatbot")
st.info("👋 Welcome, guest! You can start chatting after uploading a PDF.")

user_email = "guest@example.com"

if "user_messages" not in st.session_state:
    st.session_state.user_messages = {}
if user_email not in st.session_state.user_messages:
    st.session_state.user_messages[user_email] = []

if "feedback" not in st.session_state:
    st.session_state.feedback = {}

if "language" not in st.session_state:
    st.session_state.language = "en"

language = st.selectbox("🌐 Choose language", ["en", "hi"])
translator = Translator()

# ---------------- TEXT INPUT ONLY -------------------
prompt = st.chat_input("💬 Ask your question from the uploaded files")

# ---------------- DISPLAY CHAT -------------------
for idx, msg in enumerate(st.session_state.user_messages[user_email]):
    with st.chat_message(msg['role']):
        display_text = translator.translate(msg['content'], dest=language).text if language != "en" else msg['content']
        st.markdown(display_text)
        if msg['role'] == 'assistant':
            col1, col2, col3 = st.columns([1, 1, 6])
            with col1:
                st.button("👍", key=f"like_{idx}")
            with col2:
                st.button("👎", key=f"dislike_{idx}")
            with col3:
                st.code(display_text, language='text')

# ---------------- FILE UPLOAD -------------------

uploaded_files = st.file_uploader("📄 Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)

@st.cache_resource(show_spinner="🔄 Indexing content...")
def create_vectorstore_from_files(pdf_files):
    try:
        loaders = []
        for file in pdf_files:
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

# ---------------- HANDLE PROMPT -------------------
if prompt:
    if not uploaded_files:
        st.warning("📌 Please upload at least one PDF to proceed.")
        st.stop()

    st.chat_message("user").markdown(prompt)
    st.session_state.user_messages[user_email].append({"role": "user", "content": prompt})

    try:
        vectorstore = create_vectorstore_from_files(uploaded_files)
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
        st.session_state.user_messages[user_email].append({"role": "assistant", "content": response})

        # ---------------- SAVE CHAT TO PDF -------------------
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        chat_lines = [
            f"{m['role'].capitalize()}: {m['content']}"
            for m in st.session_state.user_messages[user_email]
        ]
        uploaded_names = ", ".join([file.name for file in uploaded_files]) if uploaded_files else "None"
        model_used = "Groq - llama3-8b-8192"

        pdf_filename = f"chat_history_{user_email.replace('@', '_at_')}.pdf"
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        pdf.cell(0, 10, f"Chat history for {user_email}", ln=True)
        pdf.cell(0, 10, f"Generated at: {now}", ln=True)
        pdf.cell(0, 10, f"Model Used: {model_used}", ln=True)
        pdf.cell(0, 10, f"Files Uploaded: {uploaded_names}", ln=True)
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

        # ---------------- EMAIL CHAT HISTORY -------------------
        st.subheader("📧 Email Chat History")
        receiver_email = st.text_input("Enter recipient email address")
        send_email = st.button("Send Email")

        if send_email:
            sender_email = os.getenv("EMAIL_ADDRESS")
            sender_password = os.getenv("EMAIL_PASSWORD")

            msg = MIMEMultipart()
            msg["From"] = sender_email
            msg["To"] = receiver_email
            msg["Subject"] = "Your Chatbot Conversation History"
            msg.attach(MIMEText("Attached is your PDF chat history.", "plain"))

            with open(pdf_filename, "rb") as f:
                attach = MIMEApplication(f.read(), _subtype="pdf")
                attach.add_header("Content-Disposition", "attachment", filename=pdf_filename)
                msg.attach(attach)

            with smtplib.SMTP("smtp.gmail.com", 587) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                server.sendmail(sender_email, receiver_email, msg.as_string())

            st.success(f"📧 Chat history emailed to {receiver_email}")

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")









