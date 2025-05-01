import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import pickle
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

TEXT_CHUNKS_PATH = "text_chunks.pkl"
FAISS_INDEX_PATH = "faiss_index"

def get_pdf_text(pdf_docs):
    """Extract text from multiple PDFs."""
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text + "\n"
        except Exception as e:
            print(f"Error reading {pdf.name}: {e}")
    return text

def get_text_chunks(text):
    """Split extracted text into clean, manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
    chunks = text_splitter.split_text(text)
    # Clean and filter chunks
    cleaned_chunks = [chunk.strip() for chunk in chunks if isinstance(chunk, str) and chunk.strip()]
    return cleaned_chunks

def save_data(text_chunks):
    with open(TEXT_CHUNKS_PATH, "wb") as f:
        pickle.dump(text_chunks, f)

def load_data():
    if os.path.exists(TEXT_CHUNKS_PATH):
        with open(TEXT_CHUNKS_PATH, "rb") as f:
            return pickle.load(f)
    return None

def get_vector_store(text_chunks):
    """Generate FAISS vector store from text chunks safely."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Check for invalid or oversized chunks
    safe_chunks = []
    for i, chunk in enumerate(text_chunks):
        if len(chunk) <= 3000:
            safe_chunks.append(chunk)
        else:
            print(f"Chunk {i} skipped (too long: {len(chunk)} chars)")

    if not safe_chunks:
        raise ValueError("No valid chunks to embed. Check your PDF content.")

    try:
        vector_store = FAISS.from_texts(safe_chunks, embedding=embeddings)
        vector_store.save_local(FAISS_INDEX_PATH)
    except Exception as e:
        print("Error creating vector store:", e)
        raise

def get_conversational_chain():
    """Setup Gemini-powered QA chain."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the context, say: "Answer is not available in the context." Do not guess.

    Context:\n{context}\n
    Question:\n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    try:
        db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Failed to load vector index: {e}")
        return

    docs = db.similarity_search(user_question)
    chain = get_conversational_chain()
    try:
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        st.write("Reply:", response["output_text"])
    except Exception as e:
        st.error(f"Failed to generate response: {e}")

def main():
    st.set_page_config(page_title="Chat with PDF using Gemini 💁")
    st.header("Chat with PDFs 💬")

    user_question = st.text_input("Ask a question from the PDF files:")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("📂 Upload & Process PDFs")
        pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing... ⏳"):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                if text_chunks:
                    save_data(text_chunks)
                    get_vector_store(text_chunks)
                    st.success("✅ Processing complete!")
                else:
                    st.error("❌ No valid text extracted from the PDFs.")

if __name__ == "__main__":
    main()
