"""
rag_engine.py
Core RAG (Retrieval-Augmented Generation) Pipeline:
- Document ingestion (.txt, .pdf)
- Whitespace normalization and chunking
- Embeddings generation (Sentence-Transformers: all-MiniLM-L6-v2)
- FAISS vector storage (IndexFlatIP for cosine similarity)
- Top-k context retrieval
- LLM answer generation (OpenAI gpt-4o-mini) with extractive fallback
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from openai import OpenAI


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------
@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    text: str
    source: str
    chunk_id: int

    @property
    def source_file(self) -> str:
        """Alias for source for convenience."""
        return self.source


# ---------------------------------------------------------------------------
# RAG Prompt Templates (Strict vs Hybrid ChatGPT Mode)
# ---------------------------------------------------------------------------
RAG_PROMPT_TEMPLATE = """You are an AI Study Assistant that helps students understand their own course material. 
Answer the student's question using ONLY the context below.

Rules:
- Base your answer strictly on the provided context.
- If the context does not contain the answer, say "I couldn't find this in your uploaded material" (or the equivalent in the requested answer language) instead of guessing.
- Answer in {language}, even if the source material or question is in another language.
- Keep the answer clear, exam-oriented, and well-structured (use bullet points or short paragraphs where helpful).
- After the answer, list the source file(s) the information came from.

Context:
{context}

Student's Question: {question}

Answer (in {language}):"""

HYBRID_PROMPT_TEMPLATE = """You are an intelligent, friendly AI Study Assistant and tutor (combining student course notes with broad general AI knowledge like ChatGPT).

Guidelines:
1. If the context from the student's notes below is relevant to the question, prioritize and ground your answer in that context, and cite the source file(s) at the end.
2. If the context is empty or does not contain the answer (e.g., general science, math, coding, history, casual conversation, or conceptual explanations), answer thoroughly, accurately, and helpfully using your broad general knowledge (like ChatGPT).
3. Answer in {language}, even if the context or question is in another language.
4. Keep the explanation clear, structured (use bullet points, code blocks, or short paragraphs where helpful), engaging, and educational.

Context from Uploaded Notes (if relevant):
{context}

Student's Question: {question}

Answer (in {language}):"""



LANG_CODE_MAP = {
    "english": "en",
    "en": "en",
    "hindi": "hi",
    "hi": "hi",
    "telugu": "te",
    "te": "te",
    "tamil": "ta",
    "ta": "ta",
}


def translate_text(text: str, target_lang: str) -> str:
    """Translate text to target language using deep-translator with error fallback."""
    clean_target = LANG_CODE_MAP.get(target_lang.strip().lower(), target_lang.strip().lower()[:2])
    if clean_target == "en" or not text.strip():
        return text
    try:
        from deep_translator import GoogleTranslator
        if len(text) > 4000:
            chunks = [text[i:i+3500] for i in range(0, len(text), 3500)]
            translated = [GoogleTranslator(source="auto", target=clean_target).translate(c) for c in chunks]
            return " ".join(translated)
        return GoogleTranslator(source="auto", target=clean_target).translate(text)
    except Exception:
        return text


# ---------------------------------------------------------------------------
# Text Preprocessing & Chunking
# ---------------------------------------------------------------------------
def normalize_whitespace(text: str) -> str:
    """Clean and normalize irregular whitespaces, tabs, and newlines."""
    # Replace multiple whitespaces and newlines with single space
    cleaned = re.sub(r"\s+", " ", text)
    return cleaned.strip()


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 800,
    overlap: int = 150,
) -> List[Chunk]:
    """
    Split normalized text into ~800-character chunks with ~150-character overlap.
    Preserves word boundaries where possible.
    """
    normalized = normalize_whitespace(text)
    if not normalized:
        return []

    chunks: List[Chunk] = []
    start = 0
    text_len = len(normalized)
    chunk_id = 0

    while start < text_len:
        end = start + chunk_size
        if end >= text_len:
            # Reached end of text
            chunk_content = normalized[start:text_len].strip()
            if chunk_content:
                chunks.append(Chunk(text=chunk_content, source=source, chunk_id=chunk_id))
            break

        # Attempt to snap to word boundary within the last 60 characters of window
        split_point = normalized.rfind(" ", start, end)
        if split_point != -1 and split_point > start + (chunk_size - 80):
            end = split_point

        chunk_content = normalized[start:end].strip()
        if chunk_content:
            chunks.append(Chunk(text=chunk_content, source=source, chunk_id=chunk_id))
            chunk_id += 1

        # Advance start position with overlap
        start = max(start + 1, end - overlap)

    return chunks


# ---------------------------------------------------------------------------
# Document Ingestion (.txt and .pdf)
# ---------------------------------------------------------------------------
def extract_text_from_file(file_path: Path) -> str:
    """Extract raw text from a .txt or .pdf file."""
    ext = file_path.suffix.lower()
    
    if ext == ".txt":
        try:
            return file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return file_path.read_text(encoding="latin-1", errors="replace")

    elif ext == ".pdf":
        text_parts = []
        reader = PdfReader(str(file_path))
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        return "\n".join(text_parts)

    return ""


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------
class Embedder:
    """
    Wraps SentenceTransformer("all-MiniLM-L6-v2").
    Produces L2-normalized float32 numpy vectors so inner product equals cosine similarity.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of text strings into normalized float32 embeddings.
        Returns a 2D numpy array of shape (len(texts), embedding_dim).
        """
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.astype(np.float32)


# ---------------------------------------------------------------------------
# Vector Store (FAISS IndexFlatIP)
# ---------------------------------------------------------------------------
class VectorStore:
    """
    Wraps faiss.IndexFlatIP for cosine similarity search over normalized vectors.
    Stores chunk metadata (text, source filename, chunk id) alongside vectors.
    """

    def __init__(self):
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunks: List[Chunk] = []
        self.dimension: Optional[int] = None

    def add(self, chunks: List[Chunk], vectors: np.ndarray) -> None:
        """
        Add chunks and their normalized embedding vectors to FAISS index.
        """
        if len(chunks) == 0 or vectors.shape[0] == 0:
            return

        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)

        if self.index is None:
            self.dimension = vectors.shape[1]
            self.index = faiss.IndexFlatIP(self.dimension)

        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, top_k: int = 4) -> List[Tuple[Chunk, float]]:
        """
        Search for top_k most similar chunks.
        Returns list of (Chunk, similarity_score) sorted by descending similarity.
        """
        if self.is_empty():
            return []

        if query_vector.ndim == 1:
            query_vector = np.expand_dims(query_vector, axis=0)

        if query_vector.dtype != np.float32:
            query_vector = query_vector.astype(np.float32)

        k = min(top_k, len(self.chunks))
        distances, indices = self.index.search(query_vector, k)

        results: List[Tuple[Chunk, float]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx < len(self.chunks):
                results.append((self.chunks[idx], float(dist)))

        return results

    def is_empty(self) -> bool:
        """Check if vector store contains any indexed chunks."""
        return self.index is None or self.index.ntotal == 0

    def clear(self) -> None:
        """Reset index and chunk storage."""
        self.index = None
        self.chunks = []
        self.dimension = None


# ---------------------------------------------------------------------------
# Complete RAG Engine
# ---------------------------------------------------------------------------
class RagEngine:
    """
    Orchestrates ingestion, chunking, embedding, retrieval, and generation.
    Supports OpenAI gpt-4o-mini and extractive fallback mode.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        embedder: Optional[Embedder] = None,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.api_key = api_key.strip() if api_key and api_key.strip() else None
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        self.embedder = embedder if embedder is not None else Embedder(model_name=model_name)
        self.vector_store = VectorStore()

    def set_api_key(self, api_key: Optional[str]) -> None:
        """Update OpenAI API key dynamically."""
        cleaned = api_key.strip() if api_key and api_key.strip() else None
        self.api_key = cleaned
        self.client = OpenAI(api_key=cleaned) if cleaned else None

    def ingest_folder(self, folder: str) -> int:
        """
        Loads, chunks, embeds, and indexes all .txt and .pdf documents in folder.
        Resets existing vector store and returns total chunks indexed.
        """
        self.vector_store.clear()
        folder_path = Path(folder)
        if not folder_path.exists() or not folder_path.is_dir():
            return 0

        supported_extensions = {".txt", ".pdf"}
        all_chunks: List[Chunk] = []

        files = [
            f for f in folder_path.iterdir()
            if f.is_file() and f.suffix.lower() in supported_extensions
        ]

        # Sort files for deterministic indexing order
        files.sort(key=lambda p: p.name.lower())

        for file_path in files:
            raw_text = extract_text_from_file(file_path)
            chunks = chunk_text(raw_text, source=file_path.name)
            all_chunks.extend(chunks)

        if not all_chunks:
            return 0

        # Compute embeddings in batch
        chunk_texts = [c.text for c in all_chunks]
        vectors = self.embedder.encode(chunk_texts)

        # Store in FAISS
        self.vector_store.add(all_chunks, vectors)
        return len(all_chunks)

    def retrieve(self, question: str, top_k: int = 4) -> List[Tuple[Chunk, float]]:
        """
        Embeds the question and retrieves top_k most similar chunks from vector store.
        """
        if self.vector_store.is_empty() or not question.strip():
            return []

        query_vec = self.embedder.encode([question.strip()])
        return self.vector_store.search(query_vec, top_k=top_k)

    def answer(
        self,
        question: str,
        top_k: int = 4,
        language: str = "English",
        mode: str = "hybrid",
    ) -> Tuple[str, List[str]]:
        """
        Retrieves top-k chunks, builds prompt (hybrid ChatGPT mode or strict RAG), and generates answer.
        Falls back to extractive passages if OpenAI API key is not configured.
        Supports multi-language responses (English, Hindi, Telugu, Tamil, etc.).
        Returns: (answer_text, list_of_source_filenames)
        """
        is_hybrid = mode.lower().strip() != "strict"

        # If vector store is empty
        if self.vector_store.is_empty():
            if is_hybrid and self.client:
                # Answer like ChatGPT directly using LLM broad intelligence
                prompt = (
                    f"You are a helpful, knowledgeable AI tutor. Answer the student's question clearly, thoroughly, "
                    f"and engagingly in {language}.\n\n"
                    f"Student's Question: {question.strip()}\n\n"
                    f"Answer (in {language}):"
                )
                try:
                    completion = self.client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3,
                    )
                    return completion.choices[0].message.content.strip(), []
                except Exception:
                    pass

            msg = "I couldn't find this in your uploaded material. No documents have been indexed yet. Please upload study materials (.txt or .pdf) and click 'Build / Rebuild Knowledge Base' first."
            return (translate_text(msg, language), [])

        # Retrieve top-k chunks
        retrieved = self.retrieve(question, top_k=top_k)
        if not retrieved and not is_hybrid:
            msg = "I couldn't find this in your uploaded material"
            return (translate_text(msg, language), [])

        # Deduplicate source file names preserving order
        source_filenames: List[str] = []
        for chunk, _ in retrieved:
            if chunk.source not in source_filenames:
                source_filenames.append(chunk.source)

        # Build context string
        context_passages = []
        for i, (chunk, score) in enumerate(retrieved, start=1):
            context_passages.append(
                f"[{i}] Source: {chunk.source} (Part {chunk.chunk_id}):\n{chunk.text}"
            )
        context = "\n\n".join(context_passages)

        # Select template: Hybrid (answers random/general questions too) vs Strict
        chosen_template = HYBRID_PROMPT_TEMPLATE if is_hybrid else RAG_PROMPT_TEMPLATE
        prompt = chosen_template.format(
            context=context,
            question=question.strip(),
            language=language.strip(),
        )

        # If OpenAI API is available, generate LLM response
        if self.client:
            try:
                completion = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3 if is_hybrid else 0.2,
                )
                answer_text = completion.choices[0].message.content.strip()
                return answer_text, source_filenames
            except Exception as e:
                # If API call fails, provide diagnostic and fall back to extractive passages
                error_msg = str(e)
                fallback_header = (
                    f"> [!WARNING]\n"
                    f"> OpenAI API error: `{error_msg}`.\n"
                    f"> Showing extractive search results from your notes below:\n\n"
                )
                passages_formatted = []
                for i, (chunk, score) in enumerate(retrieved, start=1):
                    passages_formatted.append(
                        f"**Passage {i}** (Similarity: {score:.2f}) — *Source: {chunk.source}*\n\n{chunk.text}"
                    )
                combined = fallback_header + "\n\n---\n\n".join(passages_formatted)
                return combined, source_filenames

        # Extractive fallback mode (no OpenAI API key configured)
        fallback_header = (
            "*(Extractive Fallback Mode — No OpenAI API key provided)*\n\n"
            "Here are the top retrieved passages from your study material relevant to your question:\n\n"
        )
        passages_formatted = []
        for i, (chunk, score) in enumerate(retrieved, start=1):
            body_text = chunk.text
            # If target language is non-English, attempt translating excerpt
            if language.strip().lower() not in ["english", "en"]:
                body_text = translate_text(body_text, language)
            passages_formatted.append(
                f"**Passage {i}** (Cosine Similarity: {score:.2f}) — *Source: `{chunk.source}` (Part {chunk.chunk_id})*\n\n"
                f"{body_text}"
            )
        
        translated_header = translate_text(fallback_header, language) if language.strip().lower() not in ["english", "en"] else fallback_header
        extractive_answer = translated_header + "\n\n---\n\n".join(passages_formatted)
        return extractive_answer, source_filenames
