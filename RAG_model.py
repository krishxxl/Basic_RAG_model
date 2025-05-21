import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# ----------- Step 1: Load Models -----------

# Embedding model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# GPT-2 model and tokenizer
gen_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gen_model = GPT2LMHeadModel.from_pretrained('gpt2')
gen_model.eval()

# Enable GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gen_model.to(device)

# ----------- Step 2: Document Corpus -----------

documents = [
    "The capital of France is Paris.",
    "Python is a popular language for machine learning.",
    "GPT-2 is a text generation model from OpenAI.",
    "Artificial intelligence is transforming industries.",
    "The Eiffel Tower is one of the most famous landmarks in Paris.",
]

doc_embeddings = embed_model.encode(documents, convert_to_numpy=True)

# ----------- Step 3: Build FAISS Index -----------

dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)

# ----------- Step 4: Define RAG Function -----------

def rag_query(query, top_k=2, max_gen_len=100):
    # Step 1: Embed and search
    query_vec = embed_model.encode([query], convert_to_numpy=True)
    _, indices = index.search(query_vec, top_k)
    retrieved = [documents[i] for i in indices[0]]
    context = " ".join(retrieved)

    # Step 2: Create prompt
    prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"

    # Step 3: Generate answer using GPT-2
    inputs = gen_tokenizer(prompt, return_tensors="pt").to(device)
    outputs = gen_model.generate(
        **inputs,
        max_length=inputs['input_ids'].shape[1] + max_gen_len,
        num_return_sequences=1,
        do_sample=True,
        top_p=0.95,
        temperature=0.7,
        pad_token_id=gen_tokenizer.eos_token_id
    )

    output_text = gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = output_text.split("Answer:")[-1].strip()
    return answer

# ----------- Step 5: Try a Query -----------

query = "What is GPT-2?"
print("Answer:", rag_query(query))

