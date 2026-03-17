"""
CECI 工安缺失案例庫 - RAG Query API
接收文字或圖片查詢，從 Qdrant 找回最相似的工安缺失案例
"""

import os
import base64
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from qdrant_client.models import ScoredPoint

log = logging.getLogger("uvicorn.error")

# ── 設定 ──────────────────────────────────────────────
GEMINI_API_KEY   = os.environ["GEMINI_API_KEY"]
QDRANT_URL       = os.environ["QDRANT_URL"]
QDRANT_API_KEY   = os.environ["QDRANT_API_KEY"]
EMBED_MODEL      = "gemini-embedding-2-preview"
VECTOR_SIZE      = 3072
IMAGE_COLLECTION = "image_vectors"
TEXT_COLLECTION  = "text_vectors"
# ─────────────────────────────────────────────────────

gemini_client: genai.Client = None
qdrant_client: QdrantClient = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global gemini_client, qdrant_client
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    log.info("Clients initialized")
    yield
    log.info("Shutting down")


app = FastAPI(
    title="CECI 工安缺失 RAG API",
    description="以圖搜圖、以文搜圖 — 工地工安缺失案例檢索",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────

class TextSearchRequest(BaseModel):
    query: str
    collection: str = "both"   # "text", "image", "both"
    top_k: int = 5

class SearchResult(BaseModel):
    image_name: str
    description: str
    law_ref: str
    image_path: str
    score: float
    source_collection: str

class SearchResponse(BaseModel):
    results: list[SearchResult]
    query_type: str


# ── Helpers ───────────────────────────────────────────

def embed_text_sync(text: str) -> list[float]:
    response = gemini_client.models.embed_content(
        model=EMBED_MODEL,
        contents=text,
        config=types.EmbedContentConfig(output_dimensionality=VECTOR_SIZE),
    )
    return list(response.embeddings[0].values)


def embed_image_bytes_sync(image_bytes: bytes, mime_type: str) -> list[float]:
    response = gemini_client.models.embed_content(
        model=EMBED_MODEL,
        contents=types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
        config=types.EmbedContentConfig(output_dimensionality=VECTOR_SIZE),
    )
    return list(response.embeddings[0].values)


def scored_to_result(hit: ScoredPoint, collection: str) -> SearchResult:
    p = hit.payload or {}
    return SearchResult(
        image_name=p.get("image_name", ""),
        description=p.get("description", ""),
        law_ref=p.get("law_ref", ""),
        image_path=p.get("image_path", ""),
        score=round(hit.score, 4),
        source_collection=collection,
    )


def search_collection(vector: list[float], collection: str, top_k: int) -> list[SearchResult]:
    hits = qdrant_client.search(
        collection_name=collection,
        query_vector=vector,
        limit=top_k,
        with_payload=True,
    )
    return [scored_to_result(h, collection) for h in hits]


def merge_and_rank(results_a: list, results_b: list, top_k: int) -> list:
    """合併兩個 collection 的結果，依分數排序後取 top_k"""
    seen = set()
    merged = []
    for r in sorted(results_a + results_b, key=lambda x: x.score, reverse=True):
        if r.image_name not in seen:
            seen.add(r.image_name)
            merged.append(r)
        if len(merged) >= top_k:
            break
    return merged


# ── Routes ────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stats")
def stats():
    """回傳兩個 collection 的點數統計"""
    img = qdrant_client.get_collection(IMAGE_COLLECTION)
    txt = qdrant_client.get_collection(TEXT_COLLECTION)
    return {
        IMAGE_COLLECTION: img.points_count,
        TEXT_COLLECTION: txt.points_count,
    }


@app.post("/search/text", response_model=SearchResponse)
def search_by_text(req: TextSearchRequest):
    """以文字查詢相似工安缺失案例"""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query cannot be empty")
    if req.top_k < 1 or req.top_k > 50:
        raise HTTPException(status_code=400, detail="top_k must be 1-50")

    vector = embed_text_sync(req.query)

    if req.collection == "text":
        results = search_collection(vector, TEXT_COLLECTION, req.top_k)
    elif req.collection == "image":
        results = search_collection(vector, IMAGE_COLLECTION, req.top_k)
    else:  # both
        r_text  = search_collection(vector, TEXT_COLLECTION, req.top_k)
        r_image = search_collection(vector, IMAGE_COLLECTION, req.top_k)
        results = merge_and_rank(r_text, r_image, req.top_k)

    return SearchResponse(results=results, query_type="text")


@app.post("/search/image", response_model=SearchResponse)
async def search_by_image(
    file: UploadFile = File(...),
    collection: str = "both",
    top_k: int = 5,
):
    """以圖片查詢相似工安缺失案例（上傳圖片）"""
    if top_k < 1 or top_k > 50:
        raise HTTPException(status_code=400, detail="top_k must be 1-50")

    content_type = file.content_type or "image/jpeg"
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()
    vector = embed_image_bytes_sync(image_bytes, content_type)

    if collection == "text":
        results = search_collection(vector, TEXT_COLLECTION, top_k)
    elif collection == "image":
        results = search_collection(vector, IMAGE_COLLECTION, top_k)
    else:
        r_text  = search_collection(vector, TEXT_COLLECTION, top_k)
        r_image = search_collection(vector, IMAGE_COLLECTION, top_k)
        results = merge_and_rank(r_text, r_image, top_k)

    return SearchResponse(results=results, query_type="image")


@app.post("/search/image_base64", response_model=SearchResponse)
def search_by_image_base64(payload: dict):
    """以 base64 圖片查詢（適合 VLM 整合）
    Body: {"image_b64": "...", "mime_type": "image/jpeg", "collection": "both", "top_k": 5}
    """
    image_b64  = payload.get("image_b64", "")
    mime_type  = payload.get("mime_type", "image/jpeg")
    collection = payload.get("collection", "both")
    top_k      = int(payload.get("top_k", 5))

    if not image_b64:
        raise HTTPException(status_code=400, detail="image_b64 is required")

    image_bytes = base64.b64decode(image_b64)
    vector = embed_image_bytes_sync(image_bytes, mime_type)

    if collection == "text":
        results = search_collection(vector, TEXT_COLLECTION, top_k)
    elif collection == "image":
        results = search_collection(vector, IMAGE_COLLECTION, top_k)
    else:
        r_text  = search_collection(vector, TEXT_COLLECTION, top_k)
        r_image = search_collection(vector, IMAGE_COLLECTION, top_k)
        results = merge_and_rank(r_text, r_image, top_k)

    return SearchResponse(results=results, query_type="image_base64")
