import os
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. Setup paths and directories
DOCS_DIR = "documents"
if not os.path.exists(DOCS_DIR):
    os.makedirs(DOCS_DIR)
    with open(os.path.join(DOCS_DIR, "sample.txt"), "w") as f:
        f.write("Project Launch: The local RAG chatbot project was completed successfully")

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 RAG Chatbot (Groq + FAISS)")
st.write("Ask questions based on documents in the `documents/` folder.")

# Require API key from the environment (never hardcode it)
if not os.environ.get("GROQ_API_KEY"):
    st.error("GROQ_API_KEY is not set. Run: export GROQ_API_KEY=<your key> and restart.")
    st.stop()


# 2. Build and cache the vector store
@st.cache_resource
def initialize_retriever():
    txt_loader = DirectoryLoader(DOCS_DIR, glob="*.txt", loader_cls=TextLoader)
    pdf_loader = PyPDFDirectoryLoader(DOCS_DIR)
    docs = txt_loader.load() + pdf_loader.load()
    if not docs:
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store.as_retriever(search_kwargs={"k": 3})


retriever = initialize_retriever()
if retriever is None:
    st.warning("No documents found in the 'documents/' folder.")

# 3. LLM and prompt
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.3)

system_prompt = (
    "You are an expert assistant. Answer the user's question using ONLY the provided context below. "
    "If you do not know the answer or if it's not in the context, say "
    "'I cannot find that in the provided documents.' Do not make up facts.\n\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])


# 4. RAG chain (LCEL)
def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
) if retriever else None

# 5. Chat UI
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Ask something about your documents..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        if rag_chain:
            with st.spinner("Searching documents & generating answer..."):
                try:
                    answer = rag_chain.invoke(user_input)
                except Exception as e:
                    answer = f"Error calling the LLM: {e}"
        else:
            answer = "Please place text files or PDFs in the 'documents/' directory and refresh."

        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})


# Run this using streamlit run app.py and put one or more text files or PDFs in the `documents/` folder. Then ask questions about the documents.