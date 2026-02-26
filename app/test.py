from pymilvus import connections

connections.connect(
    alias="default",
    host="34.132.226.175",
    port="19530"
)

print("Connected remotely!")