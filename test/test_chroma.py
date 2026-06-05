import chromadb

client = chromadb.PersistentClient(
    path="earning_chroma/chroma_db"
)

print(client.list_collections())