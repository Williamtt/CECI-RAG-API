"""
CECI 工安缺失案例庫 - RAG Query API
接收文字或圖片查詢，從 Qdrant 找回最相似的工安缺失案例
"""

import os
import re
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
VLM_MODEL        = "gemini-2.5-flash"
VECTOR_SIZE      = 3072
IMAGE_COLLECTION = "image_vectors"
TEXT_COLLECTION  = "text_vectors"
RERANK_POOL      = 15   # 每個 collection 各取前幾名進入 rerank 候選池
# ─────────────────────────────────────────────────────

RERANK_PROMPT_TMPL = (
    "你是工地職安稽查員。請看這張施工現場照片。\n"
    "以下是從案例庫找到的候選案例描述，請選出最符合照片中工安缺失的3個案例編號。\n\n"
    "候選案例：\n{candidates}\n\n"
    "只輸出3個編號，用逗號分隔，例如：3, 7, 12\n"
    "不要其他解釋。"
)

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
    response = qdrant_client.query_points(
        collection_name=collection,
        query=vector,
        limit=top_k,
        with_payload=True,
    )
    return [scored_to_result(h, collection) for h in response.points]


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


def vlm_rerank(image_bytes: bytes, mime_type: str,
               candidates: list[SearchResult]) -> list[str]:
    """
    讓 VLM 從候選案例中選出最符合照片缺失的3個，回傳 [image_name, ...]
    """
    lines = "\n".join(
        f"[{i+1}] {r.description}" for i, r in enumerate(candidates)
    )
    prompt = RERANK_PROMPT_TMPL.format(candidates=lines)
    response = gemini_client.models.generate_content(
        model=VLM_MODEL,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            prompt,
        ],
    )
    text = response.text.strip()
    nums = [int(x) - 1 for x in re.findall(r"\d+", text) if 1 <= int(x) <= len(candidates)]
    seen: set[int] = set()
    indices: list[int] = []
    for n in nums:
        if n not in seen:
            seen.add(n)
            indices.append(n)
        if len(indices) >= 3:
            break
    # 若 VLM 未給足3個，補前幾名
    for i in range(len(candidates)):
        if len(indices) >= 3:
            break
        if i not in seen:
            seen.add(i)
            indices.append(i)
    return [candidates[i].image_name for i in indices]


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


@app.post("/search/image_rerank", response_model=SearchResponse)
async def search_by_image_rerank(
    file: UploadFile = File(...),
    top_k: int = 10,
):
    """VLM Re-ranking 高精度查詢

    流程：
    1. 圖片 embed → 從 image_vectors 取 top-15 + text_vectors 取 top-15（候選池）
    2. VLM 看照片 + 候選描述 → 選出最相關的 top-3（高信心）
    3. 其餘名次由 text_vectors 廣度搜尋補齊

    評測結果：law@1 = 17%（vs 原始圖搜圖 13%），適合用於最終判斷的主要參考案例。
    """
    if top_k < 1 or top_k > 50:
        raise HTTPException(status_code=400, detail="top_k must be 1-50")

    content_type = file.content_type or "image/jpeg"
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()
    vector = embed_image_bytes_sync(image_bytes, content_type)

    # Step 1: 建候選池（image_vectors top-15 + text_vectors top-15，去重）
    r_img = search_collection(vector, IMAGE_COLLECTION, RERANK_POOL)
    r_txt = search_collection(vector, TEXT_COLLECTION, RERANK_POOL)
    seen: set[str] = set()
    pool: list[SearchResult] = []
    for r in r_img + r_txt:
        if r.image_name not in seen:
            seen.add(r.image_name)
            pool.append(r)

    # Step 2: VLM 選出 top-3
    try:
        top3_names = vlm_rerank(image_bytes, content_type, pool)
        name_to_result = {r.image_name: r for r in pool}
        reranked = []
        for i, name in enumerate(top3_names):
            if name in name_to_result:
                r = name_to_result[name]
                reranked.append(SearchResult(
                    image_name=r.image_name,
                    description=r.description,
                    law_ref=r.law_ref,
                    image_path=r.image_path,
                    score=round(1.0 - i * 0.01, 4),
                    source_collection="reranked",
                ))
    except Exception as e:
        log.warning(f"VLM rerank failed: {e}, falling back to score-based merge")
        reranked = merge_and_rank(r_img, r_txt, 3)

    # Step 3: 補足 top_k（用 text_vectors 廣度搜尋填 4-top_k 名）
    reranked_names = {r.image_name for r in reranked}
    r_fill = search_collection(vector, TEXT_COLLECTION, top_k + len(reranked))
    fill = [r for r in r_fill if r.image_name not in reranked_names][:max(0, top_k - len(reranked))]

    results = reranked + fill
    return SearchResponse(results=results, query_type="image_rerank")
