import os

from dotenv import load_dotenv
from pathlib import Path
from llama_index import SimpleDirectoryReader
from llama_index.node_parser import SimpleNodeParser
from llama_index.llms import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index import (
    download_loader,
    ServiceContext,
    VectorStoreIndex,
    StorageContext,
    SimpleDirectoryReader
)
from llama_index.vector_stores import PineconeVectorStore
import pinecone


load_dotenv()
pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENVIRONMENT"],
)
if __name__ == "__main__":
    print("Going to ingest pinecone documentation...")
    documents = SimpleDirectoryReader("docs/txt").load_data()

    llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
    embed_model = OpenAIEmbedding(model="text-embedding-ada-002", embed_batch_size=100)
    service_context = ServiceContext.from_defaults(
        llm=llm, embed_model=embed_model
    )

    index_name = "documentation-db"
    pinecone_index = pinecone.Index(index_name=index_name)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
        service_context=service_context,
        show_progress=True,
    )
    print("finished ingesting...")
