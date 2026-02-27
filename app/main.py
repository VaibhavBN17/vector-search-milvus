from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import asyncio
from datetime import datetime
from pymilvus import connections, Collection
from app.models import insert_url_document, search_url_documents
# from app.schemas import UrlDocumentInput, TextInput
from app.schemas import UrlDocumentInput, VectorInput
# from app.embedding import generate_embedding
import threading

# -----------------------------
# CONFIG
# -----------------------------
MILVUS_HOST = "34.132.226.175"
MILVUS_PORT = "19530"
COLLECTION_NAME = "url_documents"

# -----------------------------
# FASTAPI APP
# -----------------------------
app = FastAPI()

# -----------------------------
# REQUEST COUNTERS
# -----------------------------
success_count = 0
failure_count = 0
counter_lock = threading.Lock()
# -----------------------------
# STARTUP EVENT
# -----------------------------
@app.on_event("startup")
async def startup():
    print("Connecting to Milvus...")

    # 1️⃣ Connect
    connections.connect(
        alias="default",
        host=MILVUS_HOST,
        port=MILVUS_PORT
    )

    print("Connected to Milvus.")

    # 2️⃣ Access existing collection
    collection = Collection(COLLECTION_NAME)

    # 3️⃣ Load into memory (REQUIRED for search)
    collection.load()

    app.state.collection = collection

    print(f"Collection '{COLLECTION_NAME}' loaded successfully!")


# -----------------------------
# INSERT ENDPOINT
# -----------------------------
@app.post("/insert-url")
async def insert_url(data: UrlDocumentInput, request: Request):
    collection = request.app.state.collection
    loop = asyncio.get_running_loop()

    # Generate embedding in background thread
    embedding = await loop.run_in_executor(
        None,
        generate_embedding,
        data.content
    )

    # Insert into Milvus
    insert_url_document(
        collection,
        data.content,
        data.url,
        embedding
    )

    return {"status": "Inserted successfully"}


# -----------------------------
# SEARCH ENDPOINT
# -----------------------------
# @app.post("/search-url")
# async def search_url(data: TextInput, request: Request):
#     collection = request.app.state.collection
#     loop = asyncio.get_running_loop()

#     # 1️⃣ Generate embedding
#     query_embedding = await loop.run_in_executor(
#         None,
#         generate_embedding,
#         data.text
#     )

#     # 2️⃣ Search Milvus
#     rows = await loop.run_in_executor(
#         None,
#         search_url_documents,
#         collection,
#         query_embedding,
#         10  # top_k
#     )

#     return [
#         {
#             "content": row[0],
#             "url": row[1],
#             "distance": float(row[2])
#         }
#         for row in rows
#     ]


@app.post("/search-url")
async def search_url(data: VectorInput, request: Request):
    collection = request.app.state.collection
    loop = asyncio.get_running_loop()

    # Direct vector search (NO embedding generation)
    rows = search_url_documents(
    collection,
    data.vector,
    10
)

    return [
        {
            "content": row[0],
            "url": row[1],
            "distance": float(row[2])
        }
        for row in rows
    ]

# -----------------------------
# LOGGING MIDDLEWARE
# -----------------------------
@app.middleware("http")
async def count_requests(request: Request, call_next):
    global success_count, failure_count

    try:
        response = await call_next(request)

        if response.status_code < 400:
            with counter_lock:
                success_count += 1
        else:
            with counter_lock:
                failure_count += 1

        return response

    except Exception:
        with counter_lock:
            failure_count += 1
        raise
    
@app.get("/stats")
def get_stats():
    global success_count, failure_count

    with counter_lock:
        success = success_count
        failure = failure_count
        total = success + failure

        success_count = 0
        failure_count = 0

    return {
        "success": success,
        "failure": failure,
        "total": total
    }
