
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

# ---------------- LOGIN + SIGNUP -------------------

CREDENTIALS = {
    "User8493@gmail.com": "user8493pass",
    "nishantali777@gmail.com": "nishantpass",
    "john.doe@gmail.com": "john123",
    "jane.smith@yahoo.com": "jane456",
    "info@example.com": "info789",
    "test.user@hotmail.com": "test000",
    "admin@company.com": "admin999"
}

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "signup_mode" not in st.session_state:
    st.session_state.signup_mode = False

if "reset_mode" not in st.session_state:
    st.session_state.reset_mode = False

if not st.session_state.authenticated:
    st.title("🔐 Login to Access Chatbot")
    mode = st.radio("Choose mode:", ["Login", "Sign Up", "Reset Password"])

    if mode == "Sign Up":
        new_email = st.text_input("New Email")
        new_password = st.text_input("New Password", type="password")
        if st.button("Create Account"):
            if new_email and new_password:
                email_key = new_email.strip().lower()
                if email_key in [e.lower() for e in CREDENTIALS]:
                    st.warning("⚠️ Email already exists.")
                else:
                    CREDENTIALS[new_email] = new_password
                    st.success("✅ Account created! Please login.")
                    st.experimental_rerun()
            else:
                st.warning("Please enter both email and password.")

    elif mode == "Reset Password":
        reset_email = st.text_input("Enter your registered email")
        new_reset_password = st.text_input("Enter new password", type="password")
        if st.button("Reset Password"):
            reset_email_key = reset_email.strip().lower()
            matched_email = next((e for e in CREDENTIALS if e.lower() == reset_email_key), None)
            if matched_email:
                CREDENTIALS[matched_email] = new_reset_password
                st.success("✅ Password reset successfully. Please login.")
                st.experimental_rerun()
            else:
                st.error("❌ Email not found.")

    else:
        email_input = st.text_input("Email")
        password_input = st.text_input("Password", type="password")
        if st.button("Login"):
            email_key = email_input.strip().lower()
            password = password_input.strip()
            email_matched = None
            for stored_email in CREDENTIALS:
                if stored_email.lower() == email_key:
                    email_matched = stored_email
                    break
            if email_matched and CREDENTIALS[email_matched] == password:
                st.session_state.authenticated = True
                st.session_state.user_email = email_matched
                st.experimental_rerun()
            else:
                st.error("❌ Invalid email or password")
    st.stop()

user_email = st.session_state.user_email
ALLOWED_USERS = list(CREDENTIALS.keys())
ADMIN_USERS = ["admin@company.com"]

if user_email.lower() not in [email.lower() for email in ALLOWED_USERS]:
    st.error(f"❌ Access denied for {user_email}")
    st.stop()

# ---------------- APP HEADER -------------------

st.title("🤖 Secure RAG Chatbot")
if st.button("🚪 Logout"):
    st.session_state.clear()
    st.experimental_rerun()

# ---------------- ADMIN PANEL -------------------

if user_email.lower() in [email.lower() for email in ADMIN_USERS]:
    with st.sidebar:
        st.subheader("👑 Admin Panel")
        if "user_messages" in st.session_state:
            selected_user = st.selectbox("Select user to view chat:", list(st.session_state.user_messages.keys()))
            st.write("Chat history:")
            for msg in st.session_state.user_messages[selected_user]:
                icon = "🧑" if msg['role'] == 'user' else "🤖"
                st.markdown(f"{icon} **{msg['role'].capitalize()}**: {msg['content']}")
        else:
            st.write("No messages yet.")

# ---------------- CHAT MEMORY -------------------

if "user_messages" not in st.session_state:
    st.session_state.user_messages = {}
if user_email not in st.session_state.user_messages:
    st.session_state.user_messages[user_email] = []

if "feedback" not in st.session_state:
    st.session_state.feedback = {}

for idx, msg in enumerate(st.session_state.user_messages[user_email]):
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])
        if msg['role'] == 'assistant':
            col1, col2, col3 = st.columns([1, 1, 6])
            with col1:
                if st.button("👍", key=f"like_{idx}"):
                    st.session_state.feedback[idx] = "like"
                    st.toast("✅ You liked the response.")
            with col2:
                if st.button("👎", key=f"dislike_{idx}"):
                    st.session_state.feedback[idx] = "dislike"
                    st.toast("⚠️ You disliked the response.")

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

# ---------------- CHAT INPUT -------------------

prompt = st.chat_input("💬 Ask your question from the uploaded files")

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









