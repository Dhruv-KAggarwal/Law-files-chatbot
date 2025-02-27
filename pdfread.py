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

# Paths for saving data
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
    """Split extracted text into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def save_data(text_chunks):
    """Save text chunks to a file."""
    with open(TEXT_CHUNKS_PATH, "wb") as f:
        pickle.dump(text_chunks, f)

def load_data():
    """Load saved text chunks."""
    if os.path.exists(TEXT_CHUNKS_PATH):
        with open(TEXT_CHUNKS_PATH, "rb") as f:
            return pickle.load(f)
    return None

def get_vector_store(text_chunks):
    """Generate and save FAISS vector store."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(FAISS_INDEX_PATH)

def get_conversational_chain():
    """Create a question-answering chain using Gemini AI."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the context, say: "answer is not available in the context." Do not provide incorrect answers.

    Context:\n {context}?\n
    Question:\n {question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def user_input(user_question):
    """Handle user input and generate responses."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Load FAISS index with dangerous deserialization enabled
    new_db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    print(response)
    st.write("Reply: ", response["output_text"])

def main():
    """Main function for Streamlit UI."""
    st.set_page_config(page_title="Chat with PDF using Gemini 💁")
    st.header("Chat with PDFs 💬")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("📂 Upload & Process PDFs")
        pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing... ⏳"):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                save_data(text_chunks)
                get_vector_store(text_chunks)
                st.success("✅ Processing Complete!")

    # Load and process saved data if available
    text_chunks = load_data()
    if text_chunks:
        get_vector_store(text_chunks)

if __name__ == "__main__":
    main()
