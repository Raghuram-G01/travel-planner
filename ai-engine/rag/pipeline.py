"""
JARVIS Travel Planner — RAG Pipeline
Retrieval-Augmented Generation using FAISS + sentence-transformers
No external API keys required — fully local
"""

import json
import numpy as np
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

# ─── Lazy imports for sentence-transformers & FAISS ───────────────────────────
_model = None
_faiss = None

def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def get_faiss():
    global _faiss
    if _faiss is None:
        import faiss as _faiss_lib
        _faiss = _faiss_lib
    return _faiss


# ─── Load travel data ─────────────────────────────────────────────────────────
DATA_PATH = Path(__file__).parent.parent / "data" / "travel_data.json"

def load_data() -> Dict:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ─── Document builder ─────────────────────────────────────────────────────────
def build_documents(data: Dict) -> List[Dict[str, Any]]:
    """Convert travel data into searchable text documents"""
    docs = []
    for dest in data["destinations"]:
        # Destination overview doc
        docs.append({
            "id": dest["id"],
            "type": "destination",
            "destination": dest["name"],
            "text": (
                f"{dest['name']} in {dest['state']}. {dest['description']}. "
                f"Best visited in {', '.join(dest['best_season'])}. "
                f"Average temperature: {dest['avg_temperature']}. "
                f"Tags: {', '.join(dest['tags'])}. "
                f"Budget/day: ₹{dest['budget_per_day']['budget']} (budget), "
                f"₹{dest['budget_per_day']['mid']} (mid), "
                f"₹{dest['budget_per_day']['luxury']} (luxury)."
            ),
            "raw": dest,
        })
        # Place docs
        for place in dest.get("places", []):
            docs.append({
                "id": place["id"],
                "type": "place",
                "destination": dest["name"],
                "text": (
                    f"{place['name']} in {dest['name']}. {place['description']}. "
                    f"Category: {place['category']}. "
                    f"Rating: {place['rating']}/5. "
                    f"Entry cost: ₹{place['cost']}. "
                    f"Duration: {place['duration_hours']} hours. "
                    f"Tags: {', '.join(place['tags'])}."
                ),
                "raw": place,
            })
        # Hotel docs
        for hotel in dest.get("hotels", []):
            docs.append({
                "id": hotel["id"],
                "type": "hotel",
                "destination": dest["name"],
                "text": (
                    f"{hotel['name']} in {dest['name']}. {hotel['description']}. "
                    f"Tier: {hotel['tier']}. "
                    f"Price: ₹{hotel['price_per_night']}/night. "
                    f"Rating: {hotel['rating']}/5. "
                    f"Location: {hotel.get('location', dest['name'])}. "
                    f"Amenities: {', '.join(hotel.get('amenities', []))}."
                ),
                "raw": hotel,
            })
    # Travel tips
    for i, tip in enumerate(data.get("travel_tips", [])):
        docs.append({
            "id": f"tip_{i}",
            "type": "tip",
            "destination": "general",
            "text": f"Travel tip: {tip}",
            "raw": {"tip": tip},
        })
    return docs


# ─── FAISS Vector Store ───────────────────────────────────────────────────────
class TravelVectorStore:
    def __init__(self):
        self.documents: List[Dict] = []
        self.index = None
        self.embeddings: Optional[np.ndarray] = None

    def build(self, documents: List[Dict]):
        """Embed all documents and build FAISS index"""
        self.documents = documents
        model = get_model()
        faiss = get_faiss()

        texts = [doc["text"] for doc in documents]
        print(f"[RAG] Embedding {len(texts)} documents...")
        embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
        self.embeddings = embeddings.astype(np.float32)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.embeddings)
        print(f"[RAG] FAISS index built with {self.index.ntotal} vectors (dim={dim})")

    def search(self, query: str, top_k: int = 8, destination_filter: Optional[str] = None) -> List[Dict]:
        """Semantic search for relevant travel documents"""
        model = get_model()
        faiss = get_faiss()

        query_vec = model.encode([query]).astype(np.float32)
        distances, indices = self.index.search(query_vec, top_k * 3)  # over-fetch then filter

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.documents):
                continue
            doc = self.documents[idx]
            # Optional destination filter
            if destination_filter:
                dest_lower = destination_filter.lower()
                if (doc["destination"].lower() != dest_lower and
                        dest_lower not in doc["text"].lower()):
                    continue
            results.append({**doc, "score": float(1 / (1 + dist))})
            if len(results) >= top_k:
                break

        return results

    def search_by_type(self, query: str, doc_type: str, destination: str, top_k: int = 5) -> List[Dict]:
        """Search filtered by document type"""
        all_results = self.search(query, top_k=top_k * 3, destination_filter=destination)
        typed = [r for r in all_results if r["type"] == doc_type]
        return typed[:top_k]


# ─── Singleton vector store ───────────────────────────────────────────────────
_store: Optional[TravelVectorStore] = None

def get_store() -> TravelVectorStore:
    global _store
    if _store is None:
        data = load_data()
        docs = build_documents(data)
        _store = TravelVectorStore()
        _store.build(docs)
    return _store


def get_raw_data() -> Dict:
    return load_data()
