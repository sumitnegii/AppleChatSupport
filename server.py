#!/usr/bin/env python3
"""FastAPI Backend Server for AppleSupport AI Agent Demo."""

from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agent_pipeline import (
    run_agent_pipeline,
    get_retrieval_resources,
    get_intent_classifier,
)

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm classifier, FAISS index, and embedding models on server launch."""
    try:
        get_intent_classifier()
        get_retrieval_resources()
        print("AppleSupport Agent resources pre-warmed and ready.")
    except Exception as e:
        print(f"Warning pre-warming resources: {e}")
    yield

app = FastAPI(
    title="AppleSupport AI Agent API",
    description="Production-grade AI Support Agent demo with intent classification, FAISS retrieval, grounded generation, and escalation decisions.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)


class QueryRequest(BaseModel):
    query: str = Field(..., description="Customer support message text", min_length=1)
    k: int = Field(3, description="Number of historical cases to retrieve", ge=1, le=10)
    use_remote: bool = Field(True, description="Whether to call remote LLM for grounded generation (or fallback)")


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "agent": "AppleSupport AI Agent",
        "intents_supported": 10,
        "vector_index": "FAISS IndexFlatIP (2,000 conversations, 384 dimensions)",
    }


@app.post("/api/support")
def support_endpoint(req: QueryRequest):
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        result = run_agent_pipeline(
            query=query,
            k=req.k,
            use_remote=req.use_remote,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent pipeline error: {str(e)}")


# Mount static files directory
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def serve_index():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return HTMLResponse("<h1>AppleSupport Agent UI not found</h1>", status_code=404)
    return FileResponse(str(index_file))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
