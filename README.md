# 📚 AI-Powered Study Assistant (LLMs & RAG)

A working prototype of an **AI-Powered Study Assistant** built with **Python, Streamlit, Sentence-Transformers, FAISS, and OpenAI (GPT-4o-mini)** using **Retrieval-Augmented Generation (RAG)**.

Students upload their course materials (lecture notes, textbooks, and presentation slides as `.txt` or `.pdf`). The assistant indexes these documents into a local vector database and answers questions **grounded exclusively** in that material. Answers are accurate, exam-focused, and traceable directly to source documents with zero hallucination.

---

## 🏗️ Architecture & Pipeline

```text
Student Material (.pdf / .txt)
       │
       ▼
1. Document Ingestion (pypdf, UTF-8 normalization)
       │
       ▼
2. Chunking (~800 chars, ~150 char overlap with word boundary snap)
       │
       ▼
3. Dense Embeddings (sentence-transformers: all-MiniLM-L6-v2, L2-normalized float32)
       │
       ▼
4. Vector Store (FAISS IndexFlatIP — Cosine Similarity via Dot Product)
       │
       ▼
5. Top-K Semantic Retriever (Question Vector Search)
       │
       ▼
6. Grounded Generation (OpenAI GPT-4o-mini with fallback mode)
       │
       ▼
Streamlit Interactive Chat UI with Source Citations
```

---

## 📁 Project Structure

```text
study_assistant/
├── app.py              # Streamlit chat interface, auth guard, multi-language & voice widgets
├── auth.py             # User authentication (bcrypt), login/signup, and per-user session isolation
├── rag_engine.py       # Ingestion, chunking, embeddings, FAISS vector store, multi-lang RAG answering
├── ui_strings.py       # Multi-language translation dictionaries (English, Hindi, Telugu, Tamil)
├── users.json          # Local hashed user credentials database
├── requirements.txt    # Pinned production dependencies (including bcrypt, deep-translator)
├── README.md           # Setup, usage guide, fallback mode & 8-week roadmap
└── sample_docs/
    └── sample_notes.txt # College-level Operating Systems study notes for instant demo
```

---

## 🚀 Quickstart & Setup

### 1. Prerequisites
- Python 3.10 or higher
- Git (optional)

### 2. Create and Activate Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 💻 How to Run

Launch the Streamlit web application:

```bash
streamlit run app.py
```

Streamlit will launch locally in your default web browser (typically at `http://localhost:8501`).

---

## 📖 How to Use

### 1. OpenAI API Key (Optional)
- **Generative Mode:** Enter your OpenAI API key (`sk-...`) in the sidebar to enable synthesis and exam-oriented answer generation using `gpt-4o-mini`.
- **Extractive Fallback Mode:** You can skip entering an API key! If left blank, the app runs in **Extractive Fallback Mode**, retrieving and ranking the most relevant passages from your uploaded documents alongside cosine similarity scores and file citations.

### 2. Add Study Materials
- **Option A (Custom Notes/Slides):** Use the sidebar file uploader to upload one or more `.txt` or `.pdf` files, then click **"🔨 Build / Rebuild"**.
- **Option B (Instant Demo):** Click **"⚡ Use Sample Notes"** to immediately index the included college-level Operating Systems notes (`sample_docs/sample_notes.txt`).

### 3. Ask Questions
Ask specific questions in the chat box at the bottom, such as:
- *"What are the 4 Coffman conditions required for a deadlock?"*
- *"Compare processes and threads in terms of memory and context switching overhead."*
- *"What is the difference between counting and binary semaphores?"*
- *"Explain the page fault handling sequence in demand paging."*

Each response includes a **`Sources:`** attribution line identifying the exact document(s) used to construct the answer.

---

## 🛡️ Groundedness & Anti-Hallucination Guardrails

The RAG engine enforces strict groundedness through its system prompt:
1. Answers are derived **only** from the retrieved context.
2. If the context does not contain the answer, the assistant explicitly responds:
   > *"I couldn't find this in your uploaded material"*
   rather than guessing or drawing upon external training knowledge.
3. Every answer is structured for exam preparation (concise explanations, bullet points, clean definitions).

---

## 🛠️ Extending It: 8-Week Project Curriculum

This codebase is structured as a progressive learning foundation for an 8-week applied GenAI and RAG curriculum:

| Week | Milestone | Focus Areas & Extension Tasks |
| :--- | :--- | :--- |
| **Week 1** | **Intro to GenAI & LLMs** | Foundations of Transformers, prompt engineering, token limits, zero-shot vs few-shot prompting. |
| **Week 2** | **Understanding RAG** | Theoretical foundations of dense retrieval vs sparse search (BM25), vector similarity spaces, trade-offs between parametric memory (LLM weights) and non-parametric memory (external index). |
| **Week 3** | **System Architecture** | Component contracts, asynchronous pipelines, data flow diagrams, designing reproducible ingestion pipelines. |
| **Week 4** | **Data Preprocessing & Chunking** | Experimenting with semantic chunking, recursive character splitting, sliding window overlap optimization, and handling complex PDF tables and formulas. |
| **Week 5** | **Embeddings & Vector DBs** | Compare `all-MiniLM-L6-v2` with `text-embedding-3-small` and BGE embeddings. **Extension:** Replace FAISS with persistent vector databases like **ChromaDB** or managed cloud vector stores like **Pinecone** / **Qdrant**. |
| **Week 6** | **Building Advanced RAG Pipelines** | Implement **Hybrid Search** (FAISS Dense Vectors + BM25 Sparse Keyword search via Reciprocal Rank Fusion) and add a **Cross-Encoder Re-ranker** (`cross-encoder/ms-marco-MiniLM-L-6-v2`). |
| **Week 7** | **Chat Interface & Memory** | Add **Multi-turn Conversation Memory** (using LangChain `ConversationBufferWindowMemory` or custom session history condensation) so users can ask contextual follow-up questions. |
| **Week 8** | **Testing, Evaluation & Deployment** | Implement automated RAG evaluation scripts using **RAGAS** or **TruLens** (evaluating Faithfulness, Answer Relevance, and Context Precision). Containerize using **Docker** and deploy to **Streamlit Community Cloud** or **AWS ECS**. |

### Practical Code Extensions

#### Swapping FAISS for ChromaDB
```python
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    name="study_notes",
    metadata={"hnsw:space": "cosine"}
)

# Add chunks
collection.add(
    ids=[f"{c.source}_{c.chunk_id}" for c in chunks],
    documents=[c.text for c in chunks],
    metadatas=[{"source": c.source, "chunk_id": c.chunk_id} for c in chunks],
    embeddings=vectors.tolist()
)
```

#### Adding Conversation Memory
Maintain conversational context across turns by condensing follow-up queries:
```python
condense_prompt = f"""Given the following conversation and follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}

Follow Up Input: {question}
Standalone question:"""
```

#### Automated Evaluation with RAGAS
```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevance, context_precision
from datasets import Dataset

eval_dataset = Dataset.from_dict({
    "question": test_questions,
    "contexts": retrieved_contexts,
    "answer": generated_answers,
    "ground_truth": expected_answers,
})

results = evaluate(eval_dataset, metrics=[faithfulness, answer_relevance, context_precision])
print(results)
```

---

## 📄 License
MIT License. Created for students, educators, and developers exploring Retrieval-Augmented Generation.
