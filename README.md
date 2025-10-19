# RAG Multi-Agent System

This project demonstrates a **Retrieval-Augmented Generation (RAG)** pipeline using **LangChain** and **Groq models**.  
It retrieves information from both web pages and local PDF files, embeds the content into a vector database (Chroma),  
and answers user questions using a large language model.

---

## 🚀 Features
- Load web pages and PDF documents
- Create embeddings using Hugging Face models
- Store and query documents using Chroma vector database
- Use Groq LLM for contextual Q&A with LangChain RAG pipeline

---

## 🧩 Setup Instructions

### 1. Clone or download the project
```bash
git clone <repo_url>
cd rag_multi_agent
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate    # For Linux/Mac
venv\Scripts\activate     # For Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set environment variables
Before running the app, set your Groq API key:
```bash
export GROQ_API_KEY="your_groq_api_key"
```

If you want to use your own PDF file:
```bash
export PDF_PATH="/path/to/your/document.pdf"
```

### 5. Run the script
```bash
python app.py
```

---

## 🧠 How It Works
1. Loads the model and embeddings.
2. Fetches data from web sources and local PDFs.
3. Splits text into smaller chunks.
4. Stores embeddings in a Chroma database.
5. Builds a RAG pipeline to retrieve relevant context and generate responses.

---

## 🧾 Output Example
When you run the script, you’ll see a generated response like:
```
Response:
The projects discussed include CNN topics and PM Narendra Modi’s personal life story...
```

---

## 📦 Dependencies
See `requirements.txt` for the complete list of required Python packages.

---

## 🧑‍💻 Author
Developed by Siddarth — October 2025.
