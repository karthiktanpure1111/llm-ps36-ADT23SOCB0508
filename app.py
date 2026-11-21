import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Research Paper Q&A", layout="wide")
st.title("📚 Research Paper Q&A")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            if page.extract_text():
                text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def get_answer_from_llm(context, question):
    prompt = f"""
    Answer the question as detailed as possible from the provided context. If the answer is not in
    provided context, just say, "The answer is not available in the context." Don't provide the wrong answer.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0.3 , max_retries=1)
    response = model.invoke(prompt)
    if isinstance(response.content, str):
        return response.content
    return "".join(
        part.get("text", "") if isinstance(part, dict) else str(part)
        for part in response.content
    )

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        context = "\n\n".join([doc.page_content for doc in docs])
        answer = get_answer_from_llm(context, user_question)
        st.write("**Answer:**")
        st.write(answer)
    except Exception as e:
        st.error(f"Error: {e}. Make sure you upload and process a PDF first.")

# Sidebar for PDF Upload
with st.sidebar:
    st.header("Upload Document")
    uploaded_files = st.file_uploader("Upload your Research Paper (PDF)", accept_multiple_files=True, type=["pdf"])
    if st.button("Submit & Process"):
        if uploaded_files:
            with st.spinner("Processing PDF..."):
                raw_text = get_pdf_text(uploaded_files)
                if not raw_text.strip():
                    st.error("Could not extract text from this PDF. Please upload a text-based PDF.")
                else:
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Processing Complete!")
        else:
            st.error("Please upload a PDF file first.")

# Main Interface: Preset Buttons and Custom Q&A
st.markdown("### Quick Questions")
st.write("Click a button below to ask a common question about the paper:")

col1, col2, col3, col4 = st.columns(4)
question_to_ask = ""

if col1.button("Method"):
    question_to_ask = "What methodology or approach is used in this research paper?"
if col2.button("Dataset"):
    question_to_ask = "What dataset is used in this study, and how was it collected?"
if col3.button("Results"):
    question_to_ask = "What are the main results and findings of this paper?"
if col4.button("Limitations"):
    question_to_ask = "What are the limitations of this study as mentioned by the authors?"

if question_to_ask:
    st.info(f"**Question:** {question_to_ask}")
    user_input(question_to_ask)

st.markdown("---")
st.markdown("### Custom Question")
custom_q = st.text_input("Ask any other specific question about the paper:")
if custom_q:
    user_input(custom_q)