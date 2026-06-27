"""
Smart Phone Search — FastAPI Backend
=====================================
Chạy: uvicorn 03_api_server:app --reload --port 8000

Endpoints:
  GET /search?q=điện thoại pin trâu&limit=5&brand=Samsung&max_price=500
  GET /brands         → danh sách thương hiệu
  GET /health         → kiểm tra server
"""

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import chromadb
import os
from typing import Optional

# ─── Cấu hình ─────────────────────────────────────────────
CHROMA_PATH = "/Users/admin/smart-phone-search/data/chroma_db"
COLLECTION_NAME = "phones"
MODEL_NAME      = "paraphrase-multilingual-MiniLM-L12-v2"

# ─── Khởi tạo ─────────────────────────────────────────────
app = FastAPI(
    title="Smart Phone Search API",
    description="Tìm kiếm điện thoại bằng ngôn ngữ tự nhiên với Vector DB",
    version="1.0.0"
)

# Cho phép gọi từ HTML (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model và ChromaDB 1 lần duy nhất khi server khởi động
print(" Đang load model và ChromaDB...")
print(" Đường dẫn API đang tìm:", CHROMA_PATH)
print(" Đường dẫn tuyệt đối:", os.path.abspath(CHROMA_PATH))
embedding_model = SentenceTransformer(MODEL_NAME)
chroma_client   = chromadb.PersistentClient(path=CHROMA_PATH)
danh_sach = chroma_client.list_collections()
print("Danh sách collections API nhìn thấy:", danh_sach)
collection      = chroma_client.get_collection(COLLECTION_NAME)
print(f" Sẵn sàng! {collection.count()} điện thoại trong DB")


# ─── Models ───────────────────────────────────────────────
class PhoneResult(BaseModel):
    rank:       int
    score:      float
    brand:      str
    model:      str
    year:       int
    ram_gb:     float
    battery:    float
    price_usd:  float
    screen:     float
    front_cam:  float
    back_cam:   float
    processor:  str
    description: str


class SearchResponse(BaseModel):
    query:   str
    total:   int
    results: list[PhoneResult]


# ─── Endpoints ────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "phones_in_db": collection.count(),
        "model": MODEL_NAME
    }


@app.get("/brands")
def get_brands():
    """Trả về danh sách thương hiệu có trong DB"""
    # Lấy mẫu để tìm brand list
    sample = collection.get(limit=1000, include=["metadatas"])
    brands = sorted(set(m["brand"] for m in sample["metadatas"] if m.get("brand")))
    return {"brands": brands, "total": len(brands)}


@app.get("/search", response_model=SearchResponse)
def search(
    q:         str            = Query(..., description="Câu hỏi tìm kiếm", min_length=1),
    limit:     int            = Query(5,   description="Số kết quả trả về", ge=1, le=20),
    brand:     Optional[str]  = Query(None, description="Lọc theo thương hiệu (vd: Samsung)"),
    max_price: Optional[float]= Query(None, description="Giá tối đa (USD)"),
    min_price: Optional[float]= Query(None, description="Giá tối thiểu (USD)"),
    min_ram:   Optional[float]= Query(None, description="RAM tối thiểu (GB)"),
):
    # 1. Embed query
    query_vec = embedding_model.encode([q], normalize_embeddings=True).tolist()

    # 2. Build filter
    filters = []
    if brand:
        filters.append({"brand": {"$eq": brand}})
    if max_price is not None:
        filters.append({"price_usd": {"$lte": max_price}})
    if min_price is not None:
        filters.append({"price_usd": {"$gte": min_price}})
    if min_ram is not None:
        filters.append({"ram_gb": {"$gte": min_ram}})

    where = None
    if len(filters) == 1:
        where = filters[0]
    elif len(filters) > 1:
        where = {"$and": filters}

    # 3. Query ChromaDB
    results = collection.query(
        query_embeddings=query_vec,
        n_results=limit,
        where=where,
        include=["documents", "metadatas", "distances"]
    )

    # 4. Format kết quả
    phones = []
    for i in range(len(results["ids"][0])):
        meta  = results["metadatas"][0][i]
        score = round(1 - results["distances"][0][i], 4)
        doc   = results["documents"][0][i]
        phones.append(PhoneResult(
            rank        = i + 1,
            score       = score,
            brand       = meta.get("brand", ""),
            model       = meta.get("model", ""),
            year        = int(meta.get("year", 0)),
            ram_gb      = meta.get("ram_gb", 0),
            battery     = meta.get("battery", 0),
            price_usd   = meta.get("price_usd", 0),
            screen      = meta.get("screen", 0),
            front_cam   = meta.get("front_cam", 0),
            back_cam    = meta.get("back_cam", 0),
            processor   = meta.get("processor", ""),
            description = doc,
        ))

    return SearchResponse(query=q, total=len(phones), results=phones)
