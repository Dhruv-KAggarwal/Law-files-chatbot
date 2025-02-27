import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import pickle
import time
import concurrent.futures
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

# Global Constants
FAISS_INDEX_PATH = "faiss_index"
TEXT_CHUNKS_PATH = "text_chunks.pkl"

# Initialize Embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


# ✅ Multi-threaded function to extract text from multiple PDFs fast
def extract_text_from_pdf(pdf):
    text = ""
    try:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""  # Avoid NoneType errors
    except Exception as e:
        print(f"Error reading {pdf.name}: {e}")
    return text


def get_pdf_text(pdf_docs):
    text_data = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(extract_text_from_pdf, pdf_docs)
        text_data.extend(results)
    return " ".join(text_data)  # Combine all extracted texts


# ✅ Function to split extracted text into chunks for vector storage
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)


# ✅ Save & Load text chunks to avoid redundant processing
def save_text_chunks(text_chunks):
    with open(TEXT_CHUNKS_PATH, "wb") as f:
        pickle.dump(text_chunks, f)


def load_text_chunks():
    if os.path.exists(TEXT_CHUNKS_PATH):
        with open(TEXT_CHUNKS_PATH, "rb") as f:
            return pickle.load(f)
    return None


# ✅ Efficiently update FAISS vector store (avoiding reprocessing)
def update_vector_store(new_text_chunks):
    if os.path.exists(FAISS_INDEX_PATH):
        # Load existing FAISS index and append new embeddings
        existing_db = FAISS.load_local(FAISS_INDEX_PATH, embeddings)
        updated_db = FAISS.from_texts(new_text_chunks, embedding=embeddings)
        existing_db.merge_from(updated_db)
        existing_db.save_local(FAISS_INDEX_PATH)
    else:
        # If no existing FAISS index, create a new one
        vector_store = FAISS.from_texts(new_text_chunks, embedding=embeddings)
        vector_store.save_local(FAISS_INDEX_PATH)


# ✅ Conversational chain setup for AI responses
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    If the answer is not in the provided context, say: "Answer not available in the context."

    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)


# ✅ Function to handle user queries efficiently
def user_input(user_question):
    if not os.path.exists(FAISS_INDEX_PATH):
        st.error("No PDF data processed yet! Upload and process PDFs first.")
        return

    new_db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    if not docs:
        st.warning("No relevant information found in PDFs.")
        return

    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    st.write("Reply:", response["output_text"])


# ✅ Streamlit UI setup
def main():
    st.set_page_config(page_title="Chat with PDF", layout="wide")
    st.header("Chat with PDFs using Gemini AI 🚀")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("📂 Upload & Process PDFs")
        pdf_docs = st.file_uploader("Upload PDFs and click Submit", accept_multiple_files=True)
        
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                start_time = time.time()

                # Extract and process text
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                save_text_chunks(text_chunks)  # Save extracted text
                update_vector_store(text_chunks)  # Efficiently update FAISS

                st.success(f"Processing Completed ✅ (Time: {time.time() - start_time:.2f} sec)")

    # Load previously saved text chunks (if available)
    text_chunks = load_text_chunks()
    if text_chunks:
        update_vector_store(text_chunks)  # Ensure FAISS is always updated


# ✅ Ensuring script runs properly
if __name__ == "__main__":
    main()
