from openai import OpenAI
import uuid

client = OpenAI()  # Assumes API key is set in env or elsewhere

# Simple in-memory store
vector_store = {}

def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def embed_chunks(chunks):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=chunks
    )
    return [d.embedding for d in response.data]

def store_embeddings(chunks, embeddings):
    for chunk, embedding in zip(chunks, embeddings):
        vector_store[str(uuid.uuid4())] = {"embedding": embedding, "chunk": chunk}

def get_context(query, top_k=3):
    query_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=[query]
    ).data[0].embedding

    def cosine_similarity(a, b):
        from numpy import dot
        from numpy.linalg import norm
        return dot(a, b) / (norm(a) * norm(b))

    similarities = []
    for record in vector_store.values():
        score = cosine_similarity(query_embedding, record["embedding"])
        similarities.append((score, record["chunk"]))

    similarities.sort(reverse=True)
    return [chunk for _, chunk in similarities[:top_k]]