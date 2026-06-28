from langchain_huggingface import HuggingFaceEmbeddings


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

texts=[
    "Hello world",
    "I am Rony",
    "You are a good person",
]

vector = embeddings.embed_documents(texts)
print(vector)