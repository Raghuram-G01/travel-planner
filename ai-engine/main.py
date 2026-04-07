"""
JARVIS Travel Planner — FastAPI Main Entry Point
Python AI Engine: RAG + Trip Generator + Chatbot
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

from rag.ai_engine import generate_trip
from rag.chatbot import generate_chat_response
from rag.pipeline import get_store, get_raw_data

app = FastAPI(
    title="JARVIS Travel AI Engine",
    description="AI-powered travel planning with RAG + FAISS",
    version="1.0.0"
)

# CORS — allow Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request Models ───────────────────────────────────────────────────────────
class TripRequest(BaseModel):
    destination: str
    days: int
    budget: float
    preferences: List[str] = []

class ChatRequest(BaseModel):
    session_id: str
    message: str

class SearchRequest(BaseModel):
    query: str
    destination: Optional[str] = None
    top_k: int = 5


# ─── Startup: Initialize RAG (pre-warm) ──────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    print("[JARVIS] Initializing RAG pipeline...")
    try:
        store = get_store()
        print(f"[JARVIS] RAG ready — {store.index.ntotal} vectors indexed")
    except Exception as e:
        print(f"[JARVIS] RAG init warning: {e}")


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "JARVIS Travel AI online", "version": "1.0.0"}


@app.get("/health")
def health():
    try:
        store = get_store()
        return {
            "status": "healthy",
            "rag_vectors": store.index.ntotal if store.index else 0,
        }
    except:
        return {"status": "initializing"}


@app.post("/generate")
def generate_trip_endpoint(req: TripRequest):
    """Generate a full trip itinerary using AI + RAG"""
    if req.days < 1 or req.days > 30:
        raise HTTPException(400, "Days must be between 1 and 30")
    if req.budget < 500:
        raise HTTPException(400, "Budget must be at least ₹500")

    result = generate_trip(
        destination=req.destination,
        days=req.days,
        budget=req.budget,
        preferences=req.preferences,
    )
    return result


@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    """Conversational travel assistant with context memory"""
    if not req.message.strip():
        raise HTTPException(400, "Message cannot be empty")

    response = generate_chat_response(
        session_id=req.session_id,
        user_message=req.message,
    )
    return response


@app.post("/search")
def search_endpoint(req: SearchRequest):
    """Semantic RAG search on travel knowledge base"""
    store = get_store()
    results = store.search(
        query=req.query,
        top_k=req.top_k,
        destination_filter=req.destination,
    )
    return {
        "query": req.query,
        "results": results,
        "total": len(results),
    }


@app.get("/destinations")
def list_destinations():
    """List all available destinations"""
    raw = get_raw_data()
    return {
        "destinations": [
            {
                "id": d["id"],
                "name": d["name"],
                "state": d["state"],
                "description": d["description"],
                "tags": d["tags"],
                "budget_per_day": d["budget_per_day"],
                "best_season": d["best_season"],
            }
            for d in raw["destinations"]
        ]
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
