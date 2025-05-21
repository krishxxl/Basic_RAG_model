# 🧠 basic_RAG_model

This repository contains a basic implementation of a **Retrieval-Augmented Generation (RAG)** system using a local **GPT-2** language model. The purpose of this project was to experiment with the foundational concepts behind RAG pipelines — combining document retrieval with language generation — and understand how they work at a low level.

## 📌 Project Purpose

I created this project as a way to:
- Explore how RAG-based architectures function
- Learn how to retrieve relevant chunks from documents using vector similarity
- Understand how small LLMs like GPT-2 behave when used for answering questions based on context

The entire pipeline is kept simple to stay close to the core mechanics, without relying on external APIs.

## 🛠️ Features

- 📄 PDF ingestion and text extraction
- 🧩 Chunking and embedding using `sentence-transformers`
- 🔍 Similarity search using FAISS
- 🧠 Text generation using a locally loaded GPT-2 model
- ❓ Hardcoded query: **"What is GPT-2?"**
- 🖥️ Interactive Streamlit frontend

## 🔧 Stack

- `transformers` (GPT-2 for generation)
- `sentence-transformers` (for dense embeddings)
- `faiss` (for fast vector similarity search)
- `streamlit` (for frontend)
- `PyPDF2` (for PDF parsing)

## 🧪 How It Works

1. **Document Loading**: A sample PDF is uploaded and converted to plain text.
2. **Chunking**: The text is split into smaller passages.
3. **Embedding**: Each passage is embedded into a vector using a sentence transformer.
4. **Querying**: A hardcoded query `"What is GPT-2?"` is embedded and used to retrieve the most relevant chunks from the document.
5. **Generation**: The retrieved chunks are used as context in a prompt passed to GPT-2 to generate an answer.
6. **Display**: The answer is displayed in a simple Streamlit UI.

## 📝 Notes

- The query is hardcoded to `"What is GPT-2?"` — this keeps the model focused and simplifies testing.
- GPT-2 is not trained for factual accuracy; it's primarily used here to test the idea of context-aware generation.

## 🚀 Future Improvements

- Replace GPT-2 with a more capable open-source model (e.g. T5, GPT-Neo, Mistral)
- Allow user-defined questions
- Add confidence scoring
- Expand to handle multiple document types

## 📚 Learnings

This project helped me understand:
- How document retrieval systems like RAG combine information retrieval with generation
- The importance of prompt design and chunk relevance
- The limits of small models like GPT-2 in fact-based Q&A


---

> ⚠️ This project is a learning experiment and not intended for production use.

