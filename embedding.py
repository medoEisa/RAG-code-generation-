import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder
from load_data import load

def init_chroma(db_path="./chroma_db", model_name="all-MiniLM-L6-v2"):
    """
    Initialize or load a ChromaDB collection for storing embeddings.
    """
    client = chromadb.PersistentClient(path=db_path)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
    
    try:
        collection = client.get_collection("humaneval")
        print("✅ Loaded existing ChromaDB collection.")
    except Exception:
        collection = client.create_collection("humaneval", embedding_function=ef)
        print("🆕 Created new ChromaDB collection.")
    return collection

collection = init_chroma()

def store_embeddings(collection):
    """
    Store task prompts and solutions into ChromaDB if not already stored.
    """
    data = load()
    existing = collection.count()

    if existing == 0:
        for i, (task_id, text, solution) in enumerate(zip(data["task_id"], data["prompt"], data["canonical_solution"])):
            collection.add(
                ids=[str(i)],
                documents=[text],
                metadatas=[{"task_id": task_id, "solution": solution}]
            )
        print(f"✅ Stored {len(data)} documents in ChromaDB.")
    else:
        print(f"ℹ️ ChromaDB already contains {existing} entries — skipping embedding.")

def retrieve_similar(collection, query, top_k=5, rerank=False):
    """
    Retrieve top similar code examples based on query.
    Optionally rerank results using a cross-encoder.
    """
    results = collection.query(query_texts=[query], n_results=top_k * 2 if rerank else top_k)
    docs = results["documents"][0]
    metas = results["metadatas"][0]

    if rerank:
        cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        pairs = [[query, d] for d in docs]
        scores = cross_encoder.predict(pairs)
        ranked = sorted(zip(docs, metas, scores), key=lambda x: x[2], reverse=True)[:top_k]
        docs, metas, _ = zip(*ranked)

    return docs, metas

if __name__ == "__main__":
    store_embeddings(collection)
