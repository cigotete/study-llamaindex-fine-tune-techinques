from dotenv import load_dotenv
import os

from llama_index.callbacks import LlamaDebugHandler, CallbackManager
from llama_index.chat_engine.types import ChatMode

load_dotenv()
import streamlit as st
import pinecone
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.vector_stores import PineconeVectorStore

print("***Streamlit LlamaIndex Documentation Helper***")


@st.cache_resource(show_spinner=False)
def get_index() -> VectorStoreIndex:
    pinecone.init(
        api_key=os.environ["PINECONE_API_KEY"],
        environment=os.environ["PINECONE_ENVIRONMENT"],
    )
    pinecone_index = pinecone.Index(index_name="documentation-db")
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    callback_manager = CallbackManager(handlers=[llama_debug])
    service_context = ServiceContext.from_defaults(callback_manager=callback_manager)

    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store, service_context=service_context
    )


index = get_index()
if "chat_engine" not in st.session_state.keys():
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode=ChatMode.CONTEXT, verbose=True
    )


st.set_page_config(
    page_title="Chat about doc content.",
    page_icon="🦙",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)

st.title("Chat with docs 💬")

if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me a question about document",
        }
    ]

# Draws input text box and when user hits enter, appends message to messages
if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

# Draws messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If the last message was from the user, then the assistant responds
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Here is where the assistant responds based on the user's input
            response = st.session_state.chat_engine.chat(message=prompt)
            #nodes = [node for node in response.source_nodes]
            st.write(response.response)
            # Appends the response to the messages
            message = {"role": "assistant", "content": response.response}
            # response is appended to messages
            st.session_state.messages.append(message)
