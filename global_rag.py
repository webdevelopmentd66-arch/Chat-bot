import os
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from groq import Groq
import streamlit as st
from dotenv import load_dotenv

# Local system par .env read karne ke liye
load_dotenv()

# 1. Securely load API Keys (Code mein hardcode nahi karna)
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY") or os.getenv("PINECONE_API_KEY")

if not GROQ_API_KEY or not PINECONE_API_KEY:
    raise ValueError("⚠️ GROQ_API_KEY ya PINECONE_API_KEY missing hai! Check .env file ya Streamlit Secrets.")

groq_client = Groq(api_key=GROQ_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# Free local embedding model (Outputs 384-dim vectors)
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

INDEX_NAME = "medical-rag-index"

# Pinecone Index Check/Creation
if INDEX_NAME not in [idx.name for idx in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.index(INDEX_NAME)


# 2. PDF Processing & Vector Ingestion
def ingest_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    filename = os.path.basename(pdf_path)
    vectors = []

    for page_num in range(len(doc)):
        text = doc[page_num].get_text().strip()
        if len(text) < 30:
            continue

        # Simple chunking per page / paragraph
        chunks = [text[i:i + 500] for i in range(0, len(text), 450)]

        for idx, chunk in enumerate(chunks):
            # Generate Dense Embedding
            vector_values = embedder.encode(chunk).tolist()
            chunk_id = f"{filename}_p{page_num + 1}_c{idx}"

            vectors.append({
                "id": chunk_id,
                "values": vector_values,
                "metadata": {
                    "text": chunk,
                    "source": filename,
                    "page": page_num + 1
                }
            })

    # Batch upsert to Cloud Vector DB
    index.upsert(vectors=vectors)


# 3. Semantic Retrieval & Llama-3 Generation
def ask_global_rag(query):
    # Vectorize incoming user query
    query_vector = embedder.encode(query).tolist()

    # Retrieve top 3 semantic matches from Pinecone
    search_response = index.query(
        vector=query_vector,
        top_k=3,
        include_metadata=True
    )

    retrieved_texts = []
    citations = []

    for match in search_response.matches:
        if match.score >= 0.35:  # Similarity threshold
            retrieved_texts.append(match.metadata['text'])
            citations.append(f"{match.metadata['source']} (Page {match.metadata['page']})")

    if not retrieved_texts:
        return "Sorry, no relevant medical guidelines were found in the uploaded documents.", []

    # Construct Prompt for Groq (Llama-3)
    context_str = "\n---\n".join(retrieved_texts)
    prompt = f"""You are a professional medical assistant. Answer the user question STRICTLY based on the provided context below.
If the answer is not contained in the context, say "Information not found in reference materials."

Context:
{context_str}

User Question: {query}
Answer:"""

    # Call Free Llama-3 via Groq
    chat_completion = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama3-8b-8192",
        temperature=0.2
    )

    answer = chat_completion.choices[0].message.content
    return answer, list(set(citations))