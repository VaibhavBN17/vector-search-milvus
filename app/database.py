from pymilvus import connections, Collection

MILVUS_HOST = "34.132.226.175"
MILVUS_PORT = "19530"
COLLECTION_NAME = "url_documents"

# 1️⃣ Connect
connections.connect(
    alias="default",
    host=MILVUS_HOST,
    port=MILVUS_PORT
)

print("Connected to Milvus")

# 2️⃣ Access existing collection
collection = Collection(COLLECTION_NAME)

# 3️⃣ Load into memory (IMPORTANT for search)
collection.load()

print(f"Collection '{COLLECTION_NAME}' loaded successfully!")