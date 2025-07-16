import streamlit as st

st.title("RAG CHATBOT")
st.write("👋 Welcome to the chatbot. Type something below.")
#setup a seesion states varible to hold all the old message
if 'messages' not in st.session_state:
    st.session_state.messages=[]

#dispaly all the historical message
for message in st.session_state.message:
    st.chat_message(message['role']).markdown(message["content"])
prompt = st.chat_input("Pass your prompt here")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({'role':'user','content':prompt})
    response="i am your assistants"
    st.session_state.messages.append({'role':'assistance','content':prompt})




    mport warnings
import logging

import streamlit as st

# Phase 2 libraries
import os
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain_groq import ChatGroq

qroq_chat = ChatGroq(api_key="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", model_name="Llama3-8b-8192")


# Disable warnings and info logs
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

st.title('Ask Chatbot!')
# Setup a session state variable to hold all the old messages
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display all the historical messages
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

prompt = st.chat_input('Pass your prompt here')

if prompt:
    st.chat_message('user').markdown(prompt)
    # Store the user prompt in state
    st.session_state.messages.append({'role':'user', 'content': prompt})
# Phase 2 
    groq_sys_prompt = ChatPromptTemplate.from_template("""You are very smart at everything, you always give the best, 
                                            the most accurate and most precise answers. Answer the following Question: {user_prompt}.
                                            Start the answer directly. No small talk please""")

    model="llama3-8b-8192"
    qroq_chat=ChatGroq(
        groq_api_key= os.environ.get("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"),
        model_name=model
    )
    chain = groq_sys_prompt | groq_chat | StrOutputParser()
    response = chain.invoke({"user_prompt": prompt})
    response = "I am your assistant"
    st.chat_message('assistant').markdown(response)
    st.session_state.messages.append(
            {'role':'assistant', 'content':response})
    






    import warnings
import logging
import streamlit as st
import os

from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Phase 3 libraries
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA

# Load environment variables from .env file (optional)
load_dotenv()

# Suppress warnings
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# --- Streamlit App ---
st.title('Ask Chatbot!')

# Store messages
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display past messages
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

# Phase 3 (Pre-requisite)
@st.cache_resource
def get_vectorstore():
    pdf_name = "./reflexion.pdf"
    loaders = [PyPDFLoader(pdf_name)]
    # Create chunks, aka vector database–Chromadb
    index = VectorstoreIndexCreator(
        embedding=HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2'),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    ).from_loaders(loaders)
    return index.vectorstore


# Chat input
prompt = st.chat_input('Pass your prompt here')

if prompt:
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    # Prompt Template
    groq_sys_prompt = ChatPromptTemplate.from_template("""
    You are very smart at everything, you always give the best, most accurate, and most precise answers.
    Answer the following Question: {user_prompt}. Start the answer directly. No small talk please.
    """)

    # Setup Groq Chat Model
    groq_api_key = os.getenv("GROQ_API_KEY") or "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"
    qroq_chat = ChatGroq(api_key=groq_api_key, model_name="Llama3-8b-8192")

    # Chain the prompt to model to output parser
    chain = groq_sys_prompt | qroq_chat | StrOutputParser()

    # Get response
    response = chain.invoke({"user_prompt": prompt})

    # Show assistant message
    st.chat_message('assistant').markdown(response)
    st.session_state.messages.append({'role': 'assistant', 'content': response})






#code of 16/7/2025
import os
import warnings
import logging
import streamlit as st

from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA



# Disable warnings and logs
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

st.title('chat  with Chatbot!')

# Store session messages
if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

# Cache PDF vectorstore
@st.cache_resource
def get_vectorstore():
    pdf_path = "bio.pdf"
    if not os.path.exists(pdf_path):
        st.error("PDF file not found.")
        return None

    loader = PyPDFLoader(pdf_path)
    index = VectorstoreIndexCreator(
        embedding=HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2'),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    ).from_loaders([loader])
    return index.vectorstore

# Get user prompt
prompt = st.chat_input('Pass your prompt here')

if prompt:
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    try:
        # Set up Groq model
        groq_api_key = os.getenv("GROQ_API_KEY") or "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"
        groq_chat = ChatGroq(api_key=groq_api_key, model_name="llama3-8b-8192")

        # Load vectorstore
        vectorstore = get_vectorstore()
        if vectorstore is None:
            raise Exception("Vectorstore could not be loaded.")

        # Set up retrieval QA chain
        chain = RetrievalQA.from_chain_type(
            llm=groq_chat,
            chain_type='stuff',
            retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
            return_source_documents=True
        )

        result = chain({"query": prompt})
        response = result["result"]

        st.chat_message('assistant').markdown(response)
        st.session_state.messages.append({'role': 'assistant', 'content': response})

    except Exception as e:
        st.error(f"Error: {str(e)}")



import os
import warnings
import logging
import streamlit as st

from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA

# ---------------- AUTH -------------------

user = st.experimental_user
ALLOWED_USERS = ["nishant@example.com", "user1@example.com", "test@example.com"]
ADMIN_USERS = ["nishantali777@gmail.com"]

if user is None:
    st.warning("Please log in to use the chatbot.")
    st.stop()

if user.email not in ALLOWED_USERS:
    st.error(f"Access denied for {user.email}")
    st.stop()

st.title('🤖 Secure RAG Chatbot')
st.info(f"Logged in as: {user.email}")

# --------------- ADMIN PANEL ---------------
if user.email in ADMIN_USERS:
    st.sidebar.subheader("👑 Admin Panel")
    st.sidebar.write("You are logged in as an admin.")

    if "user_messages" in st.session_state:
        selected_user = st.sidebar.selectbox("Select user to view chat:", list(st.session_state.user_messages.keys()))
        st.sidebar.write("Chat history:")
        for msg in st.session_state.user_messages[selected_user]:
            role = "🧑" if msg['role'] == "user" else "🤖"
            st.sidebar.markdown(f"{role} **{msg['role'].capitalize()}**: {msg['content']}")
    else:
        st.sidebar.write("No messages yet.")

# --------------- CHATBOT LOGIC ----------------

# Disable warnings
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# Per-user message history
if "user_messages" not in st.session_state:
    st.session_state.user_messages = {}

if user.email not in st.session_state.user_messages:
    st.session_state.user_messages[user.email] = []

# Show message history
for msg in st.session_state.user_messages[user.email]:
    st.chat_message(msg['role']).markdown(msg['content'])

# Load and cache PDF
@st.cache_resource
def get_vectorstore():
    pdf_path = "bio.pdf"
    if not os.path.exists(pdf_path):
        st.error("PDF file not found.")
        return None

    loader = PyPDFLoader(pdf_path)
    index = VectorstoreIndexCreator(
        embedding=HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2'),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    ).from_loaders([loader])
    return index.vectorstore

# Chat input
prompt = st.chat_input("Ask a question from the PDF")

if prompt:
    st.chat_message('user').markdown(prompt)
    st.session_state.user_messages[user.email].append({'role': 'user', 'content': prompt})

    try:
        groq_api_key = os.getenv("GROQ_API_KEY") or "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"
        groq_chat = ChatGroq(api_key=groq_api_key, model_name="llama3-8b-8192")

        vectorstore = get_vectorstore()
        if vectorstore is None:
            raise Exception("Vectorstore could not be loaded.")

        chain = RetrievalQA.from_chain_type(
            llm=groq_chat,
            chain_type='stuff',
            retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
            return_source_documents=True
        )

        result = chain({"query": prompt})
        response = result["result"]

        st.chat_message('assistant').markdown(response)
        st.session_state.user_messages[user.email].append({'role': 'assistant', 'content': response})

    except Exception as e:
        st.error(f"Error: {str(e)}")





import os
import warnings
import logging
import streamlit as st

from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA

# ---------------- GOOGLE LOGIN AUTH -------------------

user = st.experimental_user  # Streamlit will auto-login user via Google

# Allowed user emails
ALLOWED_USERS = ["nishant@example.com", "user1@example.com", "test@example.com"]
ADMIN_USERS = ["nishant@example.com"]

# Block unauthenticated users
if user is None:
    st.warning("🔐 Please log in with your Google account to use this chatbot.")
    st.stop()

# Block unauthorized users
if user.email.lower() not in [email.lower() for email in ALLOWED_USERS]:
    st.error(f"🚫 Access denied for {user.email}")
    st.stop()

st.title("🤖 Secure RAG Chatbot")
st.success(f"✅ Logged in as: {user.email}")
if st.button("🚪 Logout"):
    st.session_state.clear()
    st.experimental_rerun()


# ----------------- ADMIN PANEL -------------------

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

# ----------------- PDF Upload + Chat Logic -------------------

# Per-user chat memory
if "user_messages" not in st.session_state:
    st.session_state.user_messages = {}
if user.email not in st.session_state.user_messages:
    st.session_state.user_messages[user.email] = []

# Show chat history
for msg in st.session_state.user_messages[user.email]:
    st.chat_message(msg['role']).markdown(msg['content'])

# PDF upload
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

# Chat prompt input
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

        groq_api_key = os.getenv("GROQ_API_KEY") or "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"
        groq_chat = ChatGroq(api_key=groq_api_key, model_name="llama3-8b-8192")

        chain = RetrievalQA.from_chain_type(
            llm=groq_chat,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )

        result = chain({"query": prompt})
        response = result["result"]

        st.chat_message("assistant").markdown(response)
        st.session_state.user_messages[user.email].append({"role": "assistant", "content": response})

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")