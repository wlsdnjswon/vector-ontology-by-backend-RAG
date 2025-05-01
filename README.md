# Vector Ontology by Backend RAG

## Overview

The **Vector Ontology by Backend RAG** repository provides a powerful framework for managing, searching, and utilizing vector-based embeddings and ontological data for advanced applications, such as natural language processing and semantic search. The system integrates components like ChromaDB for vector storage and SPARQL-based RDF retrieval to handle ontological relationships.

This solution is ideal for building hybrid systems leveraging large language models (LLMs) and structured knowledge representations.

---

## Features

### 1. Vector Store Management
- Uses **ChromaDB** for managing vector embeddings.
- Supports operations like creating or loading vector stores and performing similarity searches.
- Provides utilities for formatting retrieved documents for LLM context.

### 2. Ontology-based Retrieval
- Implements RDF-based data retrieval using SPARQL queries.
- Facilitates relationship discovery and structured information extraction.

### 3. LLM Integration
- Integrates with OpenAI's API to generate responses and manage conversations.
- Supports hybrid search combining vector similarity and ontological relationships.

---

## Installation

### Prerequisites
- Python 3.8+
- OpenAI API key for LLM integration.
- Dependencies listed in `requirements.txt`.

### Steps

1. **Clone the repository:**
   ~~~bash
   git clone https://github.com/wlsdnjswon/vector-ontology-by-backend-RAG.git
   ~~~

2. **Navigate to the project directory:**
   ~~~bash
   cd vector-ontology-by-backend-RAG
   ~~~

3. **Install the required dependencies:**
   ~~~bash
   pip install -r requirements.txt
   ~~~

---

## Usage

### Starting the System
1. **Configure the settings in `config.py`:**
   - Set the root folder path for RDF and document files.
   - Specify ChromaDB parameters and collection names.
2. **Run the main application:**
   ~~~bash
   python app.py
   ~~~

### Example Queries
- **Perform a vector search:**
   ~~~python
   vector_manager.search_similar_documents("example query")
   ~~~

- **Retrieve structured RDF data for a specific URI:**
   ~~~python
   rdf_retriever.get_all_related_info(person_uri)
   ~~~

---

## Project Structure

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

---

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your enhancements.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.
