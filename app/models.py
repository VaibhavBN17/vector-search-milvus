from pymilvus import Collection


# ---------------------------------
# INSERT DOCUMENT
# ---------------------------------
def insert_url_document(collection: Collection, content, url, embedding):
    """
    Insert one document into Milvus
    """

    data = [
        [content],      # content column
        [url],          # url column
        [embedding]     # 384-dim vector
    ]

    collection.insert(data)

    # ⚠️ DO NOT flush() every insert in production.
    # It kills performance under high QPS.
    # Milvus auto-flushes periodically.


# ---------------------------------
# SEARCH DOCUMENTS
# ---------------------------------
def search_url_documents(collection: Collection, query_embedding, limit=5):

    search_params = {
        "metric_type": "COSINE",   # ✅ MUST match index metric
        "params": {
            "ef": 64               # better recall than 32
        }
    }

    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=limit,
        output_fields=["content", "url"]
    )

    formatted_results = []

    for hits in results:
        for hit in hits:
            formatted_results.append(
                (
                    hit.entity.get("content"),
                    hit.entity.get("url"),
                    float(hit.distance)
                )
            )

    return formatted_results
