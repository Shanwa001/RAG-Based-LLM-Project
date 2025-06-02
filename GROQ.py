import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
from dotenv import load_dotenv
import os
from groq import Groq

# Load environment variables
load_dotenv()

CHROMA_DB_DIR = "chroma_db"

# Custom LLM class to wrap Groq LLaMA3
class GroqLLM(LLM):
    def _call(self, prompt, stop=None):
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        completion = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1024,
        )
        return completion.choices[0].message.content

    @property
    def _llm_type(self):
        return "groq_llama3"

# PDF -> Text
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Text -> Chunks
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

# Chunks -> Vector Store with HuggingFace Embeddings
def get_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma.from_texts(chunks, embedding=embeddings, persist_directory=CHROMA_DB_DIR)
    vectordb.persist()

# Create QA Chain
def get_qa_chain(retriever):
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
Answer the question using the context below. If the answer is not in the context, say 'Answer not found in the context.'

Context: {context}

Question: {question}

Answer:
        """
    )
    llm = GroqLLM()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type_kwargs={"prompt": prompt})
    return qa_chain

# Get answer from user question
def user_input(question):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
    retriever = vectordb.as_retriever()
    qa_chain = get_qa_chain(retriever)
    result = qa_chain.run(question)
    st.write("🧠 Reply:", result)

# Streamlit UI
def main():
    st.set_page_config(page_title="PDF Chat with LLaMA3", layout="wide")
    st.header("📄 Chat with PDF using Groq's LLaMA 3-70B 🦙⚡")

    user_question = st.text_input("Ask a Question based on the uploaded PDFs:")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("📂 Upload PDFs")
        pdf_docs = st.file_uploader("Upload PDF files", accept_multiple_files=True)
        if st.button("🚀 Submit & Process"):
            with st.spinner("Processing PDFs..."):
                text = get_pdf_text(pdf_docs)
                chunks = get_text_chunks(text)
                get_vector_store(chunks)
                st.success("✅ PDF processing complete!")

if __name__ == "__main__":
    main()
