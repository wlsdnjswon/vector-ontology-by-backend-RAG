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

### 3. Hybrid RAG and LLM Integration
- Combines context retrieved from both the to work with a separate frontend application.

**See the frontend in action (using this backend):**

**[https://vector-ontology-by-frontend-rag-git-main-wlsdnjswons-projects.vercel vector store and the ontology graph.
- Integrates with **OpenAI's API** (via `llm_handler..app/](https://vector-ontology-by-frontend-rag-git-main-wlsdnjswons-projects.vercel.app/)**

(This backend needs to be running and publicly accessible for the frontendpy`) to generate informed responses based on the retrieved context and conversation history.
- Manages conversation history to provide more demo to function correctly.)

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
- Python 3   Facilitates finding specific entities, their attributes, and relationships.

### 3. Hybrid Context Generation
-   .8+
- An **OpenAI API key** set as an environment variable (`OPENAI_API_KEY`).Combines results from both vector search and ontology retrieval to create a rich context for the LLM.

### 4. LL
- Dependencies listed in `requirements.txt` (including `Flask`, `Flask-Cors`, `rdflib`, `M Integration & Conversation Management
-   Integrates with OpenAI's API (via `llm_handler.py`) to generateopenai`, `langchain`, `chromadb`, etc.).

### Steps

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/jione-RGB/vector-ontology-by-backend- context-aware responses.
-   Maintains conversation history to provide coherent and relevant answers over multiple turns.

### 5.RAG.git 
    # Replace with your actual repository URL if different
    ```

2.  **Navigate to the project directory:**
    ```bash
    cd vector-ontology-by-backend-RAG
    ``` Flask API Server
-   Provides a `/chat` endpoint (via `app.py`) to receive user messages and return generated responses, suitable for integration with web frontends.

---

## Installation

### Prerequisites
-   Python 3.8

3.  **(Optional but Recommended) Create and activate a virtual environment:**
    ```bash
    python -m venv+
-   An **OpenAI API key** set as an environment variable (`OPENAI_API_KEY`). venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    
-   Dependencies listed in `requirements.txt`.

### Steps

1.  **Clone the repository:**
    ``````

4.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    bash
    git clone https://github.com/jione-RGB/vector-ontology-by-backend-```

5.  **Set the OpenAI API Key:**
    Make sure the `OPENAI_API_KEY` environment variableRAG.git 
    # Replace with your actual repository URL if different
    cd vector-ontology-by- is set in your terminal session or system environment before running the application.
    *   Linux/macOS: `export OPENAIbackend-RAG
    ```

2.  **Set up a virtual environment (Recommended):**
    ```bash
    _API_KEY='your_api_key'`
    *   Windows (cmd): `set OPENAI_API_KEY=your_api_key`
    *   Windows (PowerShell): `$env:OPENAIpython -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -_API_KEY='your_api_key'`

---

## Usage

### Starting the API Server
1.  **r requirements.txt
    ```

4.  **Prepare Data:**
    *   Place your RDF ontology file (e.g., `04-23-Last_noAb.rdf`) inside the `ontologybyRConfigure settings (if necessary):**
    *   Review paths and parameters in `vectorbyRAG/config.py`.AG` folder. Ensure the path in `app.py` ( `RDF_FILE = "./ontologybyRAG/
    *   Ensure your ontology file exists at the path specified in `app.py` (`RDF_FILE`)...."` ) matches.
    *   Place the documents (e.g., PDFs) you want to index into
    *   Ensure document files (e.g., PDFs) are located in the path specified in `vectorbyRAG/ the folder specified by `ROOT_FOLDER_PATH` in `vectorbyRAG/config.py`.

5config.py` (`ROOT_FOLDER_PATH`).
2.  **Run the Flask application:**
    ```.  **Set Environment Variable:**
    *   Set your OpenAI API key as an environment variable:
        ```bashbash
    python app.py
    ```
    The API server will start, typically on `http://localhost:5000`. It will initialize the ontology retriever, vector store, and LLM handler.

### Inter
        export OPENAI_API_KEY='your_openai_api_key' 
        # On Windows use `set OPENAI_API_KEY=your_openai_api_key` (cmd) 
        # oracting with the API
-   The primary way to interact is through the `/chat` endpoint using POST requests with a `$env:OPENAI_API_KEY='your_openai_api_key'` (PowerShell)
        ```

 JSON body like `{"message": "Your question here"}`.
-   This API is designed to be used by a frontend application, such as the one available at [https://vector-ontology-by-frontend-rag-git----

## Usage

### Starting the API Server
1.  **Configure settings (if needed):**
    *   Checkmain-wlsdnjswons-projects.vercel.app/](https://vector-ontology-by- `vectorbyRAG/config.py` for `ROOT_FOLDER_PATH`, ChromaDB settings, etc.
    *   Verify the `RDF_FILE` path in `app.py`.
2.  **Runfrontend-rag-git-main-wlsdnjswons-projects.vercel.app/).

### Standalone Module Usage (Example)
You can also use the retriever and manager modules programmatically:
```python
# the Flask application:**
    ```bash
    python app.py
    ```
    The API server will typically Example (ensure initialization is done or adapt as needed)
# from ontologybyRAG.rdf_retriever import SimpleRdf start on `http://localhost:5000`.

### Interacting with the API
-   The primaryInfoRetriever
# from vectorbyRAG.vector_store_manager import VectorStoreManager

# rdf_retriever = way to interact is by sending POST requests to the `/chat` endpoint with a JSON body like `{"message": "Your question here"}`.
-   This API is intended to be called by a frontend application (like the one linked above SimpleRdfInfoRetriever("./ontologybyRAG/your_ontology.rdf")
# vector_manager = VectorStoreManager("your_openai_key")
# vector_manager.create_or_load_store(...)).
-   You can also test the endpoint using tools like `curl` or Postman.

---

## Project # Load/create store first

# results = vector_manager.search_similar_documents("example query")
# uri, name = rdf_retriever.find_person_uri("Question about Jinwon Jeong")
# if Structure

```plaintext
vector-ontology-by-backend-RAG/
├── vectorbyRAG/
│   ├── document_processor.py    # Processes and chunks documents
│   ├── vector_store_manager.py  # Manages vector operations
│   ├── config.py                # Configuration settings
├── ontologybyRAG/
│   ├── rdf_retriever.py         # Handles RDF-based retrieval
│   ├── llm_handler.py           # Manages LLM interactions
│   ├── 04-23-Last_noAb.rdf      # Example Ontology file
├── app.py                       # Entry point for the application
├── requirements.
└── ...

## License

This project is licensed under the MIT License
