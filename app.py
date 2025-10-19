import os
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

def setup_environment():
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")
    if not os.environ["GROQ_API_KEY"]:
        print("Warning: GROQ_API_KEY is not set. Model may not function properly.")

def load_model():
    return init_chat_model("openai/gpt-oss-20b", model_provider="groq")

def create_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

def create_vector_store(embeddings):
    return Chroma(
        collection_name="example_collection",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db",
    )

def load_documents():
    web_loader = WebBaseLoader(web_paths=[
        "https://www.ibm.com/think/topics/convolutional-neural-networks",
        "https://www.pmindia.gov.in/en/personal_life_story/personal-life-story/"
    ])
    web_docs = web_loader.load()

    pdf_path = os.getenv("PDF_PATH", "Siddarth.pdf")
    pdf_docs = []
    if os.path.exists(pdf_path):
        pdf_loader = PyPDFLoader(pdf_path)
        pdf_docs = pdf_loader.load()
    else:
        print(f"Warning: PDF file '{pdf_path}' not found, skipping.")

    return web_docs + pdf_docs

def split_and_store_docs(docs, vector_store):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    _ = vector_store.add_documents(documents=chunks)

def build_rag_chain(vector_store, model):
    retriever = vector_store.as_retriever()
    template = """Answer the question based only on the following context:
{context}

Question: {input}
"""
    prompt = ChatPromptTemplate.from_template(template)
    rag_chain = (RunnableParallel({ "context": retriever, "input": RunnablePassthrough() }) | prompt | model)
    return rag_chain

def main():
    setup_environment()
    print("Initializing model and embeddings...")
    model = load_model()
    embeddings = create_embeddings()
    vector_store = create_vector_store(embeddings)

    print("Loading and processing documents...")
    docs = load_documents()
    split_and_store_docs(docs, vector_store)

    print("Building RAG chain and generating response...")
    rag_chain = build_rag_chain(vector_store, model)
    response = rag_chain.invoke("Give me the projects information?")
    print("\nResponse:\n", response.content)

if __name__ == "__main__":
    main()
