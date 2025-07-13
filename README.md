# SDLC Q&A Assistant: Context-Aware RAG Chatbot for Software Development Teams Using LangChain

An enterprise-grade AI chatbot that answers contextual, domain-specific questions about the Software Development Lifecycle (SDLC) using Retrieval-Augmented Generation (RAG). Built using the LangChain framework, ChromaDB, Cross-Encoder reranking, and OpenAI/HuggingFace LLMs, this repository contains two progressively advanced chatbot implementations for SDLC knowledge automation one being a basic RAG with Cross-Encoder Reranking and another with both Cross-Encoder Reranking and Contextual Memory.

## Objective

In today’s fast-paced IT environments, quick access to accurate and contextual information about SDLC processes is critical for:

- Developers  
- Project Managers  
- QA Engineers  
- DevOps Teams

This project introduces an AI-powered assistant that leverages RAG and LangChain to provide precise and contextual answers to questions related to:

- Agile, Waterfall and other methodologies  
- Software testing strategies  
- SDLC stages and documentation

## Project Variants

This repository contains two implementations of the RAG-based chatbot, each showcasing a different level of contextual awareness and performance.

### Project 1: Basic RAG with Cross-Encoder Reranking

- LLM: GPT-3.5-Turbo  
- RAG Type: Classic RAG with semantic search  
- Enhancement: Cross-encoder reranker improves retrieval relevance  
- Memory: Stateless (single-turn, no conversation history)  
- Use Case: Lightweight but highly accurate for standalone SDLC questions  



### Project 2: RAG with Contextual Memory + Cross-Encoder Reranking

- LLM: GPT-4o-mini  
- RAG Type: Context-aware retrieval using conversation memory  
- Enhancement: 
  - Uses `ConversationBufferWindowMemory` to keep recent chat history (k=1)  
  - Resolves ambiguous or pronoun-based questions.
  - Cross-encoder reranking for improved document relevance  
- Use Case: Smarter multi-turn assistant that retains conversation flow and resolves references 

## Architecture Overview

Retrieval-Augmented Generation (RAG) combines two core capabilities:

1. Retrieval – Fetches relevant documents from a vector database (ChromaDB) using semantic search.
2. Generation – Uses an LLM to generate natural language answers grounded in the retrieved content.

### Project 1 Architecture: Basic RAG with Cross-Encoder Reranking

1. User query input  
2. Retriever (ChromaDB) fetches top 4 relevant documents based on semantic similarity  
3. Cross-encoder reranker scores documents for better relevance ranking. The top 2 best scored documents are retrieved from this.
4. Prompt is constructed with retrieved context and user question  
5. GPT-3.5-Turbo generates a concise answer strictly grounded in retrieved content  
6. No conversation history or memory is used — each query is independent


### Project 2 Architecture: RAG with Contextual Memory + Cross-Encoder Reranking

1. User query input  
2. ConversationBufferWindowMemory (k=1) keeps the most recent user-assistant interaction to provide chat history context  
3. Retriever (ChromaDB) fetches top 4 relevant documents for the current query  
4. Cross-encoder reranker re-scores and picks the top 2 relevant documents  
5. Prompt template is dynamically constructed including:  
   - Retrieved documents as context  
   - User's current question  
   - Last chat history for resolving ambiguous terms (e.g., pronouns)  
6. GPT-4o-mini generates a concise, context-aware answer leveraging both retrieved documents and conversation history  



## Deployment

The chatbot is deployed using Streamlit, providing a clean, responsive, and interactive web-based UI for users to input their queries and receive instant contextual responses. This makes the assistant easily accessible across teams without requiring deep technical knowledge.

To launch the Streamlit apps:

```bash
streamlit run main.py
```

```bash
streamlit run main_with_context.py
```

## Tech Stack

| Component         | Technology Used                          |
|------------------|------------------------------------------|
| Language Model    | OpenAI GPT-3.5-Turbo / GPT-4o-mini        |
| RAG   | LangChain + ChromaDB Vector Store       |
| Embedding Model   | Sentence Transformers / HuggingFace     |
| Vector Store          | ChromaDB (for vector storage & search)  |
| Reranker Model     | cross-encoder/ms-marco-MiniLM-L-6-v2                               |
| Frontend UI       | Streamlit                               |

## Streamlit UI Screenshots

To view the chatbot interface and how users interact with the system in real time, refer to the screenshots provided in the PDFs below:

[View Streamlit UI Screenshots - Basic RAG with Cross-Encoder Reranking](./outputScreenshots.pdf)

[View Streamlit UI Screenshots - RAG with Cross-Encoder Reranking and Contextual Memory](./outputScreenshots_contextual.pdf)

These PDFs showcase:

- The user input field for asking SDLC related questions
- Model-generated answers
- Clean and responsive Streamlit layout
- Real-time query response flow
  

## Library Versions

Below are the versions of the core libraries used in this project:

- langchain==0.3.25
- langchain-core==0.3.63
- langchain-community==0.3.25
- chromadb==1.0.12
- openai==1.93.3
- huggingface_hub==0.33.0
- PyMuPDF==1.26.1
- transformers==4.34.0
- sentence-transformers==3.4.1
- streamlit==1.45.1
- streamlit_chat==0.1.1
- bitsandbytes==0.46.0


## Contact
Anusha Chaudhuri [anusha761]
