import os
import platform
import pytesseract
import streamlit as st
import pdfplumber
import numpy as np
from PIL import Image

from dotenv import load_dotenv
load_dotenv()

# Document loaders
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, Docx2txtLoader,
    CSVLoader, UnstructuredPowerPointLoader, MergedDataLoader
)

# For text splitting
from langchain.text_splitter import RecursiveCharacterTextSplitter

# For Qdrant
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# For embeddings
from langchain_huggingface import HuggingFaceEmbeddings

# For retrieval QA
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Groq LLM
from langchain_groq import ChatGroq

st.set_page_config(page_title="Chat with Qdrant + Groq")

# --- Session State ---
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "processed_data" not in st.session_state:
    st.session_state.processed_data = None
if "data_ingested" not in st.session_state:
    st.session_state.data_ingested = False
if "messages" not in st.session_state:
    st.session_state.messages = []

# Configure Tesseract if needed
current_os = platform.system()
if current_os == "Windows":
    pytesseract.pytesseract.tesseract_cmd = "D:/Program Files/Tessaract/tesseract.exe"
elif current_os == "Linux":
    pass
else:
    st.error("Unsupported OS for Tesseract.")
    st.stop()

# Utility: OCR with Tesseract
def pil_image_to_numpy(pil_image):
    return np.array(pil_image)

def extract_text_from_pdf(pdf_path, txt_output_path):
    """OCR-based extraction for purely image-based PDFs."""
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
    with open(txt_file_path, "r", encoding="utf-8") as txt_file:
        txt_content = txt_file.read()

    if pdf_documents:
        merged_text = "".join(doc.page_content for doc in pdf_documents) + "\n" + txt_content
    else:
        merged_text = txt_content
    return merged_text

def data_ingestion(uploaded_files):
    """Load documents from user-uploaded files, merge them, chunk them."""
    if not uploaded_files:
        st.error("No files to process.")
        return []

    if not st.session_state.data_ingested:
        loaders = []
        with st.spinner("Processing data..."):
            for uploaded_file in uploaded_files:
                file_path = os.path.join(os.getcwd(), uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                ext = file_path.lower()
                if ext.endswith(".pdf"):
                    # Attempt normal PDF parse
                    try:
                        pdf_loader = PyPDFLoader(file_path, extract_images=True)
                        pdf_documents = pdf_loader.load()
                    except ValueError as e:
                        st.error(f"Error loading PDF: {e}")
                        pdf_documents = None

                    # Tesseract-based fallback
                    txt_output_path = file_path.replace(".pdf", "_extracted.txt")
                    extract_text_from_pdf(file_path, txt_output_path)
                    merged_text = merge_files(pdf_documents, txt_output_path)

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

            # Combine
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

    # Split
    if st.session_state.processed_data:
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=200
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

# -----------
# Qdrant Setup
# -----------
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_vector_store(docs):
    if st.session_state.vector_store is None:
        try:
            embeddings = get_embeddings()
            client = QdrantClient(url="http://localhost:6333")

            # 1) Make sure collection exists with correct dimension
            client.recreate_collection(
                collection_name="my_collection",
                vectors_config=VectorParams(
                    size=384,  # dimension for your embedding model
                    distance=Distance.COSINE
                )
            )

            # 2) Now create the Qdrant Vector Store and add docs
            vector_store = QdrantVectorStore(
                client=client,
                collection_name="my_collection",
                embedding=embeddings,
                distance=Distance.COSINE
            )
            vector_store.add_documents(docs)

            st.session_state.vector_store = vector_store
            st.write("Vector store created in Qdrant (collection: my_collection).")
        except Exception as e:
            st.error(f"Vector store creation error: {e}")
    return st.session_state.vector_store

# -----------
# Groq LLM
# -----------
def get_groq_llm():
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        max_tokens=512,
        max_retries=2,
    )
    return llm

# -----------
# QA Retrieval
# -----------
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
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        result = qa_chain({"query": query})
        return result["result"]
    except Exception as e:
        st.error(f"Failed to generate response: {e}")
        return "I'm sorry, I couldn't process that."

def trim_chat_history(max_length=20):
    if len(st.session_state.messages) > max_length:
        st.session_state.messages = st.session_state.messages[-max_length:]

# -----------
# Main App
# -----------
def main():
    st.title("Chat with Qdrant + Groq")

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

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_question = st.chat_input("Ask a question about your documents...")
    if user_question:
        # Append user query
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        if st.session_state.vector_store is None:
            st.error("No vector store found. Please upload and process files first.")
            return

        # Groq LLM
        llm = get_groq_llm()
        response = get_response_llm(llm, st.session_state.vector_store, user_question)
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

        trim_chat_history()

if __name__ == "__main__":
    main()
