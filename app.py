# NOTE: This app.py does NOT currently run.
# It was an experimental attempt to modify our previous GPT-2 based RAG model.
# The code is a work-in-progress and may contain errors or incomplete sections.
# Use this as a reference or starting point, but it’s not production-ready yet.


import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load models
st.session_state.embed_model = st.session_state.get("embed_model", SentenceTransformer("all-MiniLM-L6-v2"))
st.session_state.tokenizer = st.session_state.get("tokenizer", GPT2Tokenizer.from_pretrained("gpt2"))
st.session_state.generator = st.session_state.get("generator", GPT2LMHeadModel.from_pretrained("gpt2").eval())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.session_state.generator.to(device)

# App title
st.title("📄🧠 PDF RAG with GPT-2")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
if uploaded_file:
    pdf_reader = PdfReader(uploaded_file)
    text_chunks = []
    for page in pdf_reader.pages:
        text = page.extract_text()
        if text:
            # Chunking by sentence size (~500 chars)
            for i in range(0, len(text), 500):
                text_chunks.append(text[i:i+500])

    st.success(f"Extracted {len(text_chunks)} chunks from PDF.")

    # Embed
    embeddings = st.session_state.embed_model.encode(text_chunks, convert_to_numpy=True)

    # Build FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    st.session_state.index = index
    st.session_state.chunks = text_chunks
    st.success("Embedding and indexing complete!")

# Query interface
query = st.text_input("Ask a question based on the PDF:")
if st.button("Generate Answer") and query:
    if "index" not in st.session_state:
        st.error("Please upload and process a PDF first.")
    else:
        # Embed query
        query_vec = st.session_state.embed_model.encode([query], convert_to_numpy=True)
        _, indices = st.session_state.index.search(query_vec, k=2)
        retrieved = [st.session_state.chunks[i] for i in indices[0]]
        context = " ".join(retrieved)

        # Build the final prompt
        prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"

        # Tokenize the input
        inputs = st.session_state.tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=1024
        ).to(device)

        # Generate response
        with torch.no_grad():
            output = st.session_state.generator.generate(
                input_ids=inputs["input_ids"],
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                eos_token_id=st.session_state.tokenizer.eos_token_id,
                pad_token_id=st.session_state.tokenizer.eos_token_id,
            )

        # Decode output
        output_text = st.session_state.tokenizer.decode(output[0], skip_special_tokens=True)

        # Extract only the generated answer (cut off the prompt)
        answer = output_text.replace(prompt, "").strip().split("\n")[0]

        # Display
        st.markdown("### 💬 Answer:")
        st.info(answer)

        if len(answer) < 5 or "¥" in answer:
            st.warning("The model gave an unclear answer. Try rephrasing your question.")


