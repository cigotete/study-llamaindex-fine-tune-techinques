from dotenv import load_dotenv
import os

from llama_index.callbacks import LlamaDebugHandler, CallbackManager

load_dotenv()
import pinecone
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.vector_stores import PineconeVectorStore

pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENVIRONMENT"]
)

if __name__ == "__main__":
    print("RAG...")

    pinecone_index = pinecone.Index(index_name="documentation-db")
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    callback_manager = CallbackManager(handlers=[llama_debug])
    service_context = ServiceContext.from_defaults(callback_manager=callback_manager)

    index = VectorStoreIndex.from_vector_store(vector_store=vector_store, service_context=service_context)

    query = "What is the content of the document?"
    query_engine = index.as_query_engine()
    response = query_engine.query(query)
    print(response)
