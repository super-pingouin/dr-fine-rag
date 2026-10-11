import json
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from chromadb.api.models.Collection import Collection

# Define project root paths dynamically based on file location
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_DB_PATH = ROOT_DIR / "chroma_db"
DEFAULT_JSON_PATH = ROOT_DIR / "data" / "raw" / "diseases.json"


def get_chroma_collection(db_path: Path = DEFAULT_DB_PATH) -> Collection :
    """Initialize ChromaDB client and return the mystery_diseases collection."""
    chroma_client = chromadb.PersistentClient(path=str(db_path))
    
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    collection = chroma_client.get_or_create_collection(
        name="mystery_diseases",
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}  # Cosine similarity for semantic text search
    )
    return collection


def sync_database(json_path: Path = DEFAULT_JSON_PATH, db_path: Path = DEFAULT_DB_PATH) -> Collection :
    """Load JSON diseases and sync them into ChromaDB (Handles upserts and deletions)."""
    collection = get_chroma_collection(db_path)
    
    # Load data
    with open(json_path, "r", encoding="utf-8") as f:
        diseases = json.load(f)

    documents, metadatas, current_json_ids = [], [], []

    for disease in diseases:
        doc_text = f"""
        Disease Name: {disease['name']}
        Visible Symptoms: {', '.join(disease['visible_symptoms'])}
        Hidden/Felt Symptoms: {', '.join(disease['hidden_symptoms'])}
        Laboratory Results: {disease['lab_results']}
        Recommended Treatment: {disease['treatment']}
        """
        documents.append(doc_text.strip())
        metadatas.append({"id": disease["id"], "name": disease["name"]})
        current_json_ids.append(disease["id"])

    current_json_set = set(current_json_ids)
    existing_db_set = set(collection.get()["ids"])

    # Clean up deleted items
    ids_to_delete = list(existing_db_set - current_json_set)
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
        print(f" Cleaned up {len(ids_to_delete)} deleted disease(s) from ChromaDB.")

    # Upsert items
    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=current_json_ids
    )
    print(f" Successfully synchronized! Total active profiles: {collection.count()}")
    return collection


def search_disease(query: str, n_results: int = 1) -> dict :
    """Search for the most relevant disease matching a player's query."""
    collection = get_chroma_collection()
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    return results


# --- Test execution block ---
if __name__ == "__main__":
    print("Running RAG synchronization test...")
    sync_database()
    
    # Test query
    test_query = "Le patient a des urines foncées et des douleurs au ventre"
    res = search_disease(test_query)
    
    print(f"\n--- Test Query Results ---")
    print(f"Query: '{test_query}'")
    print(f"Matched Disease: {res['metadatas'][0][0]['name']}")
    print(f"Snippet:\n{res['documents'][0][0][:200]}...")