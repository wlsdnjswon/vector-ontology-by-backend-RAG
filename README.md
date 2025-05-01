# Vector Ontology by Backend RAG

## Overview

The **Vector Ontology by Backend RAG** repository provides the backend API framework for an advanced RAG (Retrieval-Augmented Generation) system. It combines structured knowledge from an ontology (RDF graph via SPARQL) and unstructured information from documents (vector embeddings via ChromaDB) to generate context-rich responses using Large Language Models (LLMs).

This backend powers the interactive chatbot, enabling hybrid searches that leverage both semantic similarity and defined ontological relationships.

**Experience the chatbot powered by this backend:**

**[https://vector-ontology-by-frontend-rag-git-main-wlsdnjswons-projects.vercel.app/](https://vector-ontology-by-frontend-rag-git-main-wlsdnjswons-projects.vercel.app/)**

---

## Features

### 1. Vector Store Management (vectorbyRAG)
- Uses **ChromaDB** for efficient storage and retrieval of vector embeddings from documents.
- Supports loading documents (e.g., PDFs```markdown
# Vector Ontology by Backend RAG

## Overview

The **Vector Ontology by Backend RAG** repository provides a powerful), processing them, and performing similarity searches based on user queries.
- Provides utilities for formatting retrieved document snippets for LLM context.

### 2. Ontology-based Retrieval (ontologybyRAG)
- Implements RDF data retrieval using ** framework for managing, searching, and utilizing vector-based embeddings and ontological data for advanced applications, such as natural language processing and semantic search. The system integrates components like ChromaDB for vector storage and SPARQL-based RDF retrieval to handle ontological relationships.

ThisSPARQL** queries against an ontology file (e.g., `.rdf`).
- Identifies relevant entities (like people) in user queries and retrieves their properties and relationships from the knowledge graph.
- Formats structured ontological information for LL solution serves as the **backend API** for hybrid systems leveraging large language models (LLMs) and structured knowledge representations. It is designedM context.
---

## Features

### 1. Vector Store Management
-   Uses **Chroma coherent interactions.

### 4. Flask API Server (`app.py`)
- Provides a `/chat` APIDB** for managing vector embeddings derived from documents.
-   Supports creating/loading vector stores and performing similarity searches based on user queries.
-   Provides utilities for formatting retrieved document snippets for LLM context.

### 2. Ontology endpoint to receive user messages and return generated responses.
- Handles system initialization, request processing, and error management.
- Includes-based Retrieval
-   Implements RDF-based data retrieval from an ontology graph using SPARQL queries.
- CORS configuration for cross-origin requests from the frontend.

---

## Installation

### Prerequisites
-   Python 3.8

3.  **(Optional but Recommended) Create and activate a virtual environment:**
    ```bash
    python -m venv+
-   An **OpenAI API key** set as an environment variable (`OPENAI_API_KEY`).
-   Dependencies listed in `requirements.txt`.

## if Structure

~~~plaintext
vector-ontology-by-backend-RAG/
├── vectorbyRAG/
│   ├── vector_store_manager.py  # Manages vector operations
│   ├── config.py                # Configuration settings
├── ontologybyRAG/
│   ├── rdf_retriever.py         # Handles RDF-based retrieval
│   ├── llm_handler.py           # Manages LLM interactions
├── app.py                       # Entry point for the application
~~~

## License

This project is licensed under the MIT License
