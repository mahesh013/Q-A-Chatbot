import streamlit as st
import os
import time
from dotenv import load_dotenv

# LangChain and VectorDB Imports
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Load environment variables
load_dotenv()

# Set API keys
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
groq_api_key = os.getenv('GROQ_API_KEY')

# Initialize LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Define Prompt
prompt = ChatPromptTemplate.from_template(
    """
   Answer the question based on the provided context only.
   Please provide the most accurate response based on the question.
   <context>
   {context}
   <context>
   Question: {input}
    """
)

# Function to create vector embeddings
def create_vector_embedding():
    if "vectors" not in st.session_state:
        # Use Ollama Embeddings with Mistral
        st.session_state.embeddings = OllamaEmbeddings(model="mistral")

        # Load PDFs from correct directory path
        pdf_dir = r"E:\Generative ai\Q&A chatbot with groq\groq_chatbot\research_papers"
        st.session_state.loader = PyPDFDirectoryLoader(pdf_dir)
        st.session_state.docs = st.session_state.loader.load()

        # Split documents into chunks
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])

        # Create FAISS vector store
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# Streamlit UI
st.title("📄 AI-Powered Document Q&A")

user_prompt = st.text_input("🔍 Enter your query")

if st.button("⚡ Create Document Embeddings"):
    create_vector_embedding()
    st.success("✅ Vector Database is ready!")

if user_prompt:
    if "vectors" in st.session_state:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retriever_chain = create_retrieval_chain(retriever, document_chain)

        # Measure response time
        start = time.process_time()
        response = retriever_chain.invoke({'input': user_prompt})
        st.write(f"⏳ Response Time: {time.process_time() - start:.2f} seconds")

        # Display answer
        if 'answer' in response:
            st.subheader("💡 Answer")
            st.write(response['answer'])
        else:
            st.warning("⚠️ No relevant answer found.")

        # Document similarity search
        with st.expander("📑 Document Similarity Search"):
            if 'context' in response:
                for i, doc in enumerate(response['context']):
                    st.write(doc.page_content)
                    st.markdown("---")
            else:
                st.write("No similar documents found.")
    else:
        st.warning("⚠️ Please create the vector database first by clicking 'Create Document Embeddings'.")
