import streamlit as st
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
import os
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers.string import StrOutputParser
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'

# Initialize Streamlit App
st.title("🧠 Simple RAG Application")
st.subheader("Upload a PDF document and ask questions!")

# File upload
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# Initialize session state for retriever
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# Process the uploaded document (Only when a new file is uploaded)
if uploaded_file:
    if "uploaded_file" not in st.session_state or st.session_state.uploaded_file != uploaded_file:
        st.session_state.uploaded_file = uploaded_file
        with st.spinner("Processing document..."):
            # Save the uploaded file temporarily
            with open("uploaded_document.pdf", "wb") as f:
                f.write(st.session_state.uploaded_file.read())

            # Load the document
            loader = PyMuPDFLoader("uploaded_document.pdf")
            documents = loader.load()

            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            docs = text_splitter.split_documents(documents)

            # Embeddings and FAISS Vector Store
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            db = FAISS.from_documents(docs, embeddings)
            st.session_state.retriever = db.as_retriever()

            st.success("Document processed successfully!")

# Initialize the language model
model = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    streaming=False
)

# Define the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant developed by Mitesh. Use the provided context to answer accurately."),
        ("human", "Context: {context}"),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Create the chain with the model and prompt
chain = create_stuff_documents_chain(llm=model, prompt=prompt, output_parser=StrOutputParser())

# User input for question
query = st.text_input("Ask a question about the document:")

# Generate response
if query:
    # If document is uploaded, use retrieval-augmented generation
    if st.session_state.retriever:
        # Retrieve relevant context
        context = st.session_state.retriever.get_relevant_documents(query)
        
        # Generate and display the response
        with st.spinner("Generating answer..."):
            response = chain.invoke({"context": context, "messages": [{"role": "user", "content": query}]})
            st.write(response)

    # If no document is uploaded, handle as a general query
    else:
        no_doc_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant developed by Mitesh. Give appropriate answer."),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        
        # Simple Q&A without document context
        chain = no_doc_prompt | model | StrOutputParser()
        
        with st.spinner("Generating answer..."):
            response = chain.invoke({"messages": [{"role": "user", "content": query}]})
            st.write(response)
