import os
import streamlit as st
from dotenv import load_dotenv

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

from config import config

load_dotenv()

# API Key
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Embedding Model
embeddings = HuggingFaceEmbeddings(
    model_name=config.EMBEDDING_MODEL_NAME
)

# Vector Database
vectordb = Chroma(
    persist_directory=config.CHROMA_PERSIST_DIRECTORY,
    embedding_function=embeddings
)

# LLM
model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=400
)

# LangGraph Node
def call_model(state: MessagesState):

    system_prompt = """
    You are an academic assistant.

    Answer only from the provided context.

    If answer not found, say:
    'I could not find the answer in the documents.'
    """

    messages = [SystemMessage(content=system_prompt)] + state["messages"]

    response = model.invoke(messages)

    return {"messages": response}

# LangGraph Workflow
workflow = StateGraph(state_schema=MessagesState)

workflow.add_node("model", call_model)

workflow.add_edge(START, "model")

memory = MemorySaver()

app = workflow.compile(checkpointer=memory)

# Streamlit UI
st.set_page_config(page_title="Academic QA")

st.title("💬 Academic QA Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "chat_session"

# Show Chat History
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask your question..."):

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                # Retrieve Documents
                docs = vectordb.similarity_search(prompt, k=3)

                context = "\n\n".join(
                    [doc.page_content for doc in docs]
                )

                # RAG Prompt
                user_message = HumanMessage(
                    content=f"""
                    Context:
                    {context}

                    Question:
                    {prompt}
                    """
                )

                # Invoke Graph
                result = app.invoke(
                    {"messages": [user_message]},
                    config={
                        "configurable": {
                            "thread_id": st.session_state.thread_id
                        }
                    }
                )

                response = result["messages"][-1].content

                st.markdown(response)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response
                    }
                )

            except Exception as e:

                st.error(f"Error: {e}")