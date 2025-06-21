# SDLC-Q&A-Assistant-using-RAG-Architecture-and-LangChain-Framework

An intelligent, domain-specific chatbot designed to answer complex questions about the **Software Development Lifecycle (SDLC)** using modern **Retrieval-Augmented Generation (RAG)** architecture. Built with **LangChain**, **ChromaDB** vector store, and **LLMs from HuggingFace**, this project demonstrates a scalable and enterprise-ready solution for contextual knowledge retrieval in IT environments.

## Objective

In today’s fast-paced IT environments, quick access to accurate and contextual information about SDLC processes is critical for:

- Developers  
- Project Managers  
- QA Engineers  
- DevOps Teams

This project introduces an AI-powered assistant that leverages **RAG** and **LangChain** to provide precise and contextual answers to questions related to:

- Agile, Waterfall and other methodologies  
- Software testing strategies  
- SDLC stages and documentation

## Architecture Overview

**Retrieval-Augmented Generation (RAG)** combines two core capabilities:

1. **Retrieval** – Fetches relevant documents from a vector database (ChromaDB) using semantic search.
2. **Generation** – Uses an LLM to generate natural language answers grounded in the retrieved content.

## Deployment

The chatbot is deployed using **Streamlit**, providing a clean, responsive, and interactive web-based UI for users to input their queries and receive instant contextual responses. This makes the assistant easily accessible across teams without requiring deep technical knowledge.

To launch the Streamlit app:

```bash
streamlit run main.py
```

## Tech Stack

| Component         | Technology Used                          |
|------------------|------------------------------------------|
| Language Model    | HuggingFace Transformers (LLMs)         |
| Retrieval Layer   | LangChain + ChromaDB Vector Store       |
| Embedding Model   | Sentence Transformers / HuggingFace     |
| Database          | ChromaDB (for vector storage & search)  |
| Orchestration     | LangChain                               |
| Frontend UI       | Streamlit                               |

## Streamlit UI Screenshots

To view the chatbot interface and how users interact with the system in real time, refer to the screenshot provided in the PDF below:

📄 [View Streamlit UI Screenshots](./outputScreenshots.pdf)

This PDF showcases:

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
- huggingface_hub==0.33.0
- PyMuPDF==1.26.1
- transformers==4.34.0
- sentence-transformers==3.4.1
- streamlit==1.45.1
- streamlit_chat==0.1.1
- bitsandbytes==0.46.0


## Contact
Anusha Chaudhuri [anusha761]
