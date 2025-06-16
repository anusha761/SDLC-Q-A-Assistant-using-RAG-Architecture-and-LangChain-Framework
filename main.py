import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import torch
import re

from langchain.schema.runnable import RunnableLambda



# --- Custom LLM Wrapper ---
class CustomHuggingFacePipeline(HuggingFacePipeline):
    def invoke(self, inputs, config=None, **generation_kwargs):
        generation_kwargs.setdefault("max_new_tokens", 256)
        generation_kwargs.setdefault("temperature", 0.5)
        generation_kwargs.setdefault("do_sample", True)
        return super().invoke(inputs, config=config, **generation_kwargs)

# --- Load Model with Caching ---
@st.cache_resource
def load_local_llm():
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_4bit=True,
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
        )

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.5,
        do_sample=True,
    )
    return CustomHuggingFacePipeline(pipeline=generator)

# --- Prompt Template ---
def get_prompt_template():
    return PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a professional assistant with deep knowledge of Software Development Life Cycle (SDLC). 

Answer the user's question concisely, within 30 sentences. The user's question is given below:


{question}


Only answer questions that are strictly about SDLC concepts and models. If the question is not related to SDLC concepts and models, respond with "I don't know".
Use the provided context to answer the following question.

Context is given below:


{context}

Context ends here.

Provide your answer below.
Answer:"""
    )

# --- Build RAG Chain ---
@st.cache_resource
def build_rag_chain(_llm):
    db = Chroma(
        persist_directory="chroma_db",
        embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )
    retriever = db.as_retriever()

    # Use RunnablePassthrough for question because input is a string and both retriever & question get it
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | get_prompt_template()
        | _llm
        | RunnableLambda(lambda output: re.search(r"Answer:\s*(.*)", output, re.DOTALL).group(1).strip() if re.search(r"Answer:\s*(.*)", output, re.DOTALL) else output.strip())
    )
    return chain

# --- Load Model and Chain ---
llm = load_local_llm()
rag_chain = build_rag_chain(llm)

# --- Streamlit UI ---
st.title("SDLC Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
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
