# 🔍 AI Research Paper Intelligence System

An AI-powered semantic search engine for Machine Learning research papers. Instead of relying on exact keyword matching, the system understands the **context and meaning** of a user's query to retrieve the most relevant research papers. It also generates concise summaries and extracts important keywords, helping researchers quickly evaluate paper relevance.


---

## 🌟 Features

- 🔎 Semantic search using sentence embeddings
- 📚 Search across approximately 50,000 ArXiv Machine Learning papers
- ⚡ Fast similarity search using FAISS
- 📝 Automatic abstract summarization with DistilBART
- 🏷️ Keyword extraction using KeyBERT
- 🌐 Interactive web application built with Streamlit
- 💡 Natural language query support

---

## 🏗️ Project Workflow
 
```
                                      
                                                           User Query
                                                                │
                                                                ▼
                                                      Sentence Transformer
                                                       (all-MiniLM-L6-v2)
                                                                │
                                                                ▼
                                                       Semantic Embedding
                                                                │
                                                                ▼
                                                       FAISS Vector Search
                                                                │
                                                                ▼
                                                  Top-K Relevant Research Papers
                                                                │
                                                                ▼
                                                    ┌───────────┬────────────┬
                                                    ▼           ▼            ▼
                                               Summarizer     KeyBERT      spaCy NER
                                                    │           │            │
                                                    └───────────┴────────────┘
                                                                │
                                                                ▼
                                                        Streamlit Dashboard
```

---
## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Aditi2234/AI-Research-Paper-Intelligence-System.git
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate the Virtual Environment

```bash
.venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ⚙️ Build the Search Index

Run the following commands once to prepare the dataset and build the FAISS index.

```bash
python src/data_prep.py
python src/build_index.py
```


## ▶️ Launch the Application

Start the Streamlit application with:

```bash
streamlit run src/app.py
```

After launching, open the local URL shown in the terminal (typically **http://localhost:8501**) in your browser.

---


## 📈 Future Enhancements

Possible improvements include:

- Support for larger research collections using approximate nearest-neighbor indexing
- Filtering by publication year, author, or research category
- Citation graph visualization
- Similar-paper recommendations
- PDF preview and download support
- Deployment on Streamlit Community Cloud or Hugging Face Spaces
- User accounts with saved searches and bookmarks

---

