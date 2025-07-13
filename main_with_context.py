import streamlit as st
import re
from sentence_transformers import CrossEncoder
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
import os

# Load GPT-4o-mini Model
@st.cache_resource
def load_openai_llm():
    return ChatOpenAI(model_name="gpt-4o-mini", temperature=0.1)

# Prompt Template
def get_prompt_template():
    return PromptTemplate(
        input_variables=["context", "question", "chat_history"],
        template="""
You are a professional assistant with deep knowledge of Software Development Life Cycle (SDLC) models and methodologies.

Your job is to answer **only** questions strictly related to SDLC concepts, phases, or models.  
If the question is unrelated to SDLC, respond with: "I don't know."

Answer concisely in fewer than 10 sentences. Use retrieved context as the primary source of truth.

---

User's Question:
{question}

---

Relevant SDLC Context (from internal documents):
{context}

---

Chat History (use only to resolve ambiguity, if needed):
{chat_history}

---

Answer:
"""
    )

# Cross-Encoder Reranker
@st.cache_resource
def load_reranker():
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank_documents(query, docs, reranker, top_k=2):
    if not docs:
        st.warning("No documents retrieved for the query; skipping reranking.")
        return []
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    scored_docs = list(zip(scores, docs))
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    top_docs = [doc for _, doc in scored_docs[:top_k]]
    return top_docs

# Build Conversational Retrieval Chain with Reranking and Memory
@st.cache_resource
def build_rag_chain(_llm, _reranker, _memory):
    db = Chroma(
        persist_directory="chroma_db_sdlc_store",
        embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )
    retriever = db.as_retriever(search_kwargs={"k": 4})
    prompt = get_prompt_template()

    def format_chat_history(chat_history_str):
        return chat_history_str.strip() if chat_history_str else ""

    def retrieve_and_rerank(query, full_history=""):
        combined_query = f"{full_history}\nUser: {query}".strip()
        docs = retriever.get_relevant_documents(combined_query)
        top_docs = rerank_documents(query, docs, _reranker, top_k=2)
        if not top_docs:
            return "No relevant context found."
        return "\n\n".join([doc.page_content for doc in top_docs])

    def chain_call(inputs):
        question = inputs["question"]

        mem_data = _memory.load_memory_variables({})
        raw_chat_history = mem_data.get("chat_history", "")
        chat_history = format_chat_history(raw_chat_history)

        # Use chat history for context-aware retrieval
        context = retrieve_and_rerank(question, chat_history)

        # Still include chat history in prompt for coherence
        prompt_text = prompt.format(context=context, question=question, chat_history=chat_history)
        llm_response = _llm.invoke(prompt_text)
        return llm_response.content.strip()

    return {
        "invoke": lambda query: chain_call({
            "question": query,
            "chat_history": memory.load_memory_variables({}).get("chat_history", "")
        }),
        "save": lambda user_input, assistant_output: memory.save_context(
            {"input": user_input}, {"output": assistant_output}
        )
    }

# Initialize Components
llm = load_openai_llm()
reranker = load_reranker()
memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=False, k=1)
rag_chain = build_rag_chain(llm, reranker, memory)

# Streamlit UI
st.title("SDLC Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask a question about SDLC...")

if query:
    st.chat_message("user").markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    with st.spinner("Generating response..."):
        response = rag_chain["invoke"](query)
        rag_chain["save"](query, response)

    st.chat_message("assistant").markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
