import streamlit as st
import re
from sentence_transformers import CrossEncoder
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.chat_models import ChatOpenAI
import os

# Load GPT-3.5 Model
@st.cache_resource
def load_openai_llm():
    return ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.1)


def get_prompt_template():
    return PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a professional assistant with deep knowledge of Software Development Life Cycle (SDLC). 

Answer the user's question concisely, within 20 sentences. The user's question is given below:


{question}


Only answer questions that are strictly about SDLC concepts and models. If the question is not related to SDLC concepts and models, respond with "I don't know".
Use the provided context to answer the following question.

Context is given below:


{context}

Context ends here.

Provide your answer below.
Answer:"""
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

# RAG Chain with Reranking
@st.cache_resource
def build_rag_chain(_llm, _reranker):
    db = Chroma(
        persist_directory="chroma_db_sdlc_pdf",
        embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )
    retriever = db.as_retriever(search_kwargs={"k": 4})  # Ensure top 4 chunks are retrieved

    def retrieve_and_rerank(query):
        docs = retriever.get_relevant_documents(query)
        st.write(f"Retrieved {len(docs)} documents for query: {query}")
        top_docs = rerank_documents(query, docs, _reranker, top_k=2)
        if not top_docs:
            return "No relevant context found."
        return "\n\n".join([doc.page_content for doc in top_docs])

    # Build the chain
    chain = (
        {
            "context": RunnableLambda(retrieve_and_rerank),
            "question": RunnablePassthrough()
        }
        | get_prompt_template()
        | _llm
        | RunnableLambda(lambda output: output.content.strip())
    )
    return chain

# Load Components
llm = load_openai_llm()
reranker = load_reranker()
rag_chain = build_rag_chain(llm, reranker)

# Streamlit UI
st.title("SDLC Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
query = st.chat_input("Ask a question about SDLC...")

if query:
    st.chat_message("user").markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    with st.spinner("Generating response..."):
        response = rag_chain.invoke(query)

    st.chat_message("assistant").markdown(response.strip())
    st.session_state.messages.append({"role": "assistant", "content": response.strip()})
