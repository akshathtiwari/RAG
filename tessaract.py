import os
import platform
import pytesseract
import streamlit as st
import pdfplumber
import numpy as np
from PIL import Image

from dotenv import load_dotenv

# Load environment variables (including GROQ_API_KEY) from .env file
load_dotenv()

# --- LangChain Community Imports ---
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, Docx2txtLoader, CSVLoader,
    MergedDataLoader, UnstructuredPowerPointLoader
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# --- Groq Chat LLM ---
from langchain_groq import ChatGroq

st.set_page_config(page_title="Chat with Docs + Groq")

# --- Session State ---
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "processed_data" not in st.session_state:
    st.session_state.processed_data = None
if "data_ingested" not in st.session_state:
    st.session_state.data_ingested = False
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Tesseract Setup (optional) ---
current_os = platform.system()
if current_os == "Windows":
    # Update if your Tesseract is in a non-default path
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
elif current_os == "Linux":
    pass
else:
    st.error("Unsupported OS for Tesseract.")
    st.stop()

# -------------------------------------------------------------------------
#                          OCR & PDF Utilities
# -------------------------------------------------------------------------
def pil_image_to_numpy(pil_image):
    return np.array(pil_image)

def extract_text_from_pdf(pdf_path, txt_output_path):
    """Uses Tesseract OCR via pdfplumber to extract text from each page."""
    with pdfplumber.open(pdf_path) as pdf:
        extracted_text = ""
        for page_number, page in enumerate(pdf.pages):
            pil_image = page.to_image().original
            numpy_image = pil_image_to_numpy(pil_image)
            text = pytesseract.image_to_string(numpy_image)
            extracted_text += text + "\n"
            print(f"Processed page {page_number + 1}")

    with open(txt_output_path, "w", encoding="utf-8") as txt_file:
        txt_file.write(extracted_text)

    print(f"OCR extraction complete. Saved to {txt_output_path}")

def merge_files(pdf_documents, txt_file_path):
    """Merge text from PDFLoader (if any) with the Tesseract-extracted text."""
    with open(txt_file_path, "r", encoding="utf-8") as txt_file:
        txt_content = txt_file.read()

    if pdf_documents:
        merged_text = "".join(doc.page_content for doc in pdf_documents) + "\n" + txt_content
    else:
        merged_text = txt_content

    return merged_text

# -------------------------------------------------------------------------
#                    Data Ingestion / Splitting
# -------------------------------------------------------------------------
def data_ingestion(uploaded_files):
    """Ingests files (PDF, txt, docx, csv, pptx), merges them, 
    splits into chunks, and returns a list of docs.
    """
    if not uploaded_files:
        st.error("No files to process.")
        return []

    # Only ingest once per session
    if not st.session_state.data_ingested:
        loaders = []
        with st.spinner("Processing data..."):
            for uploaded_file in uploaded_files:
                file_path = os.path.join(os.getcwd(), uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                ext = file_path.lower()
                if ext.endswith(".pdf"):
                    try:
                        pdf_loader = PyPDFLoader(file_path, extract_images=True)
                        pdf_documents = pdf_loader.load()
                    except ValueError as e:
                        st.error(f"Error loading PDF with PyPDFLoader: {e}")
                        pdf_documents = None

                    # Extract with Tesseract
                    txt_output_path = file_path.replace(".pdf", "_extracted.txt")
                    extract_text_from_pdf(file_path, txt_output_path)
                    merged_text = merge_files(pdf_documents, txt_output_path)

                    # Save merged text
                    merged_text_path = file_path.replace(".pdf", "_merged.txt")
                    with open(merged_text_path, "w", encoding="utf-8") as merged_file:
                        merged_file.write(merged_text)

                    try:
                        txt_loader = TextLoader(merged_text_path, encoding="utf-8")
                        loaders.append(txt_loader)
                    except Exception as e:
                        st.error(f"Error loading merged text: {e}")

                elif ext.endswith(".txt"):
                    loaders.append(TextLoader(file_path, encoding="utf-8"))
                elif ext.endswith(".docx"):
                    loaders.append(Docx2txtLoader(file_path))
                elif ext.endswith(".csv"):
                    loaders.append(CSVLoader(file_path))
                elif ext.endswith(".pptx") or ext.endswith(".ppt"):
                    loaders.append(UnstructuredPowerPointLoader(file_path, mode="single"))
                else:
                    st.warning(f"Unsupported file type: {uploaded_file.name}")

            # Merge all loaders
            if loaders:
                try:
                    loader_all = MergedDataLoader(loaders=loaders)
                    documents = loader_all.load()
                    st.session_state.processed_data = documents
                    st.session_state.data_ingested = True
                    st.success("Data ingestion complete.")
                except Exception as e:
                    st.error(f"Error merging loaders: {e}")
                    return []
            else:
                st.error("No valid loaders created. Check your file types.")
                return []

    # Split if we have data
    if st.session_state.processed_data:
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=10000,
                chunk_overlap=1000
            )
            docs = text_splitter.split_documents(st.session_state.processed_data)
            st.write(f"Documents split successfully. Total chunks: {len(docs)}")
            return docs
        except Exception as e:
            st.error(f"Error splitting documents: {e}")
            return []
    else:
        st.error("No processed data available.")
        return []

# -------------------------------------------------------------------------
#                Embeddings / Vector Store (FAISS)
# -------------------------------------------------------------------------
def get_embeddings():
    """Uses a HuggingFace embedding model."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_vector_store(docs):
    """Create a FAISS vector store from docs if not already in session state."""
    if st.session_state.vector_store is None:
        try:
            embeddings = get_embeddings()
            vectorstore_faiss = FAISS.from_documents(docs, embeddings)
            vectorstore_faiss.save_local("faiss_index")
            st.session_state.vector_store = vectorstore_faiss
            st.write("Vector store created & saved locally.")
        except Exception as e:
            st.error(f"Vector store creation error: {e}")
    return st.session_state.vector_store

# -------------------------------------------------------------------------
#                  Groq LLM
# -------------------------------------------------------------------------
def get_groq_llm():
    """Returns an instance of ChatGroq. 
    Ensure GROQ_API_KEY is set in your environment or .env file."""
    llm = ChatGroq(
        model="mixtral-8x7b-32768",  # Replace with your Groq model name
        temperature=0.1,
        max_tokens=512,
        max_retries=2,
    )
    return llm

# -------------------------------------------------------------------------
#              Prompt Template & QA Helper
# -------------------------------------------------------------------------
prompt_template = """
You are a helpful assistant that uses the following context to answer a question.
If the context doesn't provide enough information, say "I don't know."
Use concise sentences.

<context>
{context}
</context>

Question: {question}

Answer:
"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

def get_response_llm(llm, vectorstore, query):
    """Given an LLM, a vector store, and a user query,
    build a RetrievalQA chain and return the answer."""
    from langchain.chains import RetrievalQA

    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        result = qa({"query": query})
        return result['result']
    except Exception as e:
        st.error(f"Failed to generate response: {e}")
        return "I'm sorry, I couldn't process that."

def trim_chat_history(max_length=20):
    """Keep the last `max_length` messages in session state."""
    if len(st.session_state.messages) > max_length:
        st.session_state.messages = st.session_state.messages[-max_length:]

# -------------------------------------------------------------------------
#                           MAIN APP
# -------------------------------------------------------------------------
def main():
    st.title("Chat with Documents (Groq Integration)")

    with st.sidebar:
        st.subheader("1) Upload & Process Files")
        uploaded_files = st.file_uploader(
            "Upload your files", accept_multiple_files=True,
            type=["pdf", "txt", "docx", "csv", "pptx", "ppt"]
        )

        if st.button("Process Files") and uploaded_files:
            docs = data_ingestion(uploaded_files)
            if docs:
                get_vector_store(docs)

    # Display existing chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_question = st.chat_input("Ask a question about your documents...")
    if user_question:
        # Store user message
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        # Check we have a vector store
        if st.session_state.vector_store is None:
            st.error("No vector store found. Please upload and process files first.")
            return

        # Create or reuse the Groq LLM
        llm = get_groq_llm()

        # Generate response
        response = get_response_llm(llm, st.session_state.vector_store, user_question)
        st.session_state.messages.append({"role": "assistant", "content": response})

        with st.chat_message("assistant"):
            st.markdown(response)

        # Trim chat history if too long
        trim_chat_history()

if __name__ == "__main__":
    main()
