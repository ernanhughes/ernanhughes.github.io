+++
date = '2025-02-23T07:57:28Z'
draft = false
title = 'Efficient Similarity Search with FAISS and SQLite in Python'
tag=['faiss', 'Vector Databases', 'Ollama', 'SmartAnswer']
categories=['faiss', 'vector', 'Ollama', 'SmartAnswer']
+++

## Summary

This is another component in `SmartAnswer` and enhanced LLM interface.

In this blog post, we introduce a wrapper class, `FaissDB`, which integrates FAISS with SQLite or any database to manage document embeddings and enable efficient similarity search. 
This approach combines FAISS's vector search capabilities with the storage and querying power of a database, making it ideal for applications such as Retrieval-Augmented Generation (RAG) and recommendation systems.

It builds up this tool [PaperSearch]({{< relref "post/paper_search.md" >}}). 

I also did an introductory post on [Faiss]({{< relref "post/faiss.md" >}}). 


## Advantages of Using FAISS

[FAISS](https://github.com/facebookresearch/faiss) provides several advantages when dealing with large-scale similarity searches:

* **High-Speed Search**: FAISS is optimized for fast nearest-neighbor search, enabling efficient retrieval of similar vectors even in massive datasets.
* **Scalability**: Designed for large-scale applications, FAISS efficiently handles millions of embeddings and performs searches in milliseconds.
* **Multiple Distance Metrics**: Supports various similarity measures, including L2 distance and inner product, making it versatile for different use cases.
* **Efficient Indexing Techniques**: FAISS offers different indexing strategies, such as IVF (Inverted File Index) and HNSW (Hierarchical Navigable Small World), allowing for further speed optimizations.
* **GPU Acceleration**: Supports GPU-based computation, significantly speeding up large-scale searches and embedding comparisons.
* **Seamless Integration**: FAISS integrates well with popular ML and AI frameworks, making it easy to use in AI-powered applications.

## Overview of the `FaissDB` Class
The `FaissDB` class is designed to handle document embeddings, perform similarity searches, and manage the FAISS index efficiently. Below are its main functionalities:

- **Building an Index**: Converts documents into embeddings and stores them in a FAISS index.
- **Searching for Similar Documents**: Retrieves the most similar documents based on a given query.
- **Storing and Retrieving Embeddings**: Saves embeddings in an SQLite database for persistence.
- **Loading Documents**: Fetches and cleans text data from the database for processing.

Let's explore each functionality in detail.

## Implementation Details

### 1. Setting Up the Database Connection
The class initializes an SQLite connection and loads configurations from `appConfig`.

```python
class FaissDB:
    def __init__(self, config=appConfig):
        self.db = Database()
        self.conn = sqlite3.connect(config["DB_NAME"])
        self.cursor = self.conn.cursor()
        self.config = config
        self.embedding_model = config["EMBEDDING_MODEL"]
        self.document_index = None
```

### 2. Generating and Storing Embeddings
FAISS relies on embeddings to perform similarity searches. We generate embeddings using Ollama’s model and store them in the database.

```python
def get_embeddings(self, documents) -> np.ndarray:
    embeddings = []
    for order, doc in enumerate(documents, start=1):
        response = ollama.embeddings(model=self.embedding_model, prompt=doc[1])
        embedding_vector = np.array(response["embedding"], dtype="float32")
        self.cursor.execute(
            """INSERT INTO document_embeddings (faiss_index_id, document_page_split_id, embedding)
            VALUES (?, ?, ?)""",
            (order, doc[0], embedding_vector.tobytes()),
        )
        embeddings.append(response["embedding"])
    self.conn.commit()
    return np.array(embeddings, dtype="float32")
```

### 3. Building the FAISS Index
Once embeddings are generated, they are stored in a FAISS index for fast retrieval.

```python
def build_index(self, embeddings: np.ndarray) -> faiss.IndexFlatIP:
    dimension = embeddings.shape[1]
    self.document_index = faiss.IndexFlatL2(dimension)
    self.document_index.add(embeddings)
    return self.document_index
```

### 4. Performing a Similarity Search
The `search` method retrieves the most relevant documents based on the query’s embedding.

```python
def search(self, text, k=appConfig["top_k"]):
    if self.document_index is None:
        raise ValueError("Index not set. Please set the index before searching.")
    query_embedding = self.get_embedding(text)
    distances, indices = self.document_index.search(query_embedding, k)
    results = [SearchResult(idx, distances[0][i], self.get_doc(idx)) for i, idx in enumerate(indices[0])]
    return results
```

### 5. Retrieving Documents from the Database
The retrieved indices are used to fetch the corresponding documents from SQLite.

```python
def get_doc(faiss_index, config=appConfig):
    conn = sqlite3.connect(config["DB_NAME"])
    cursor = conn.cursor()
    cursor.execute(
        f"""SELECT text FROM document_page_split
            WHERE id IN (
                SELECT document_page_split_id FROM document_embeddings
                WHERE faiss_index_id = {faiss_index})"""
    )
    document = cursor.fetchone()
    return document[0] if document else ""
```

### 6. Saving and Loading the Index
To avoid recomputing the FAISS index every time, we save and reload it when needed.

```python
@staticmethod
def save(index, filename):
    faiss.write_index(index, filename)

@staticmethod
def load(filename):
    return faiss.read_index(filename)
```

## Complete code

The current class looks like

```python

import sqlite3
import re
import numpy as np
import faiss
import ollama
import logging

from config import appConfig
from database import Database

logger = logging.getLogger(__name__)


class SearchResult:
    def __init__(self, index, score, doc):
        self.index = index
        self.score = score
        self.doc = doc

    def __repr__(self):
        return f"Score: {self.score}, Index: {self.index}, Doc: {self.doc[:15]} ..."

    def __str__(self):
        return self.doc

    @staticmethod
    def from_tuple(t):
        return SearchResult(t[0], t[1], t[2])

    @staticmethod
    def from_tuples(tuples):
        return [SearchResult.from_tuple(t) for t in tuples]

class FaissDB:
    def __init__(self, config=appConfig):
        self.db = Database()
        self.conn = sqlite3.connect(config["DB_NAME"])
        self.cursor = self.conn.cursor()
        self.config = config
        self.embedding_model = config["EMBEDDING_MODEL"]
        self.document_index = None

    def set_index(self, document_index):
        self.document_index = document_index

    def search(self, text, k = appConfig["top_k"]):
        if self.document_index is None:
            raise ValueError("Index not set. Please set the index before searching.")
        logger.info(f"Searching for similar items to: {text}")
        query_embedding = self.get_embedding(text)
        # Search for the top-k most similar embeddings
        distances, indices = self.document_index.search(query_embedding, k)

        results = []
        for (i, idx) in enumerate(indices[0]):
            doc = self.get_doc(idx)
            self.cursor.execute(
                """INSERT INTO search_results (query, distance, faiss_index, document_text, document_index, document_page_id) 
                VALUES (?, ?, ?, ? ,?, ?)""",
                (text, distances[0][i], idx, doc, idx, 0),
            )
            results.append(SearchResult(idx, distances[0][i], doc))
        self.conn.commit()
        return results


    def build_index(self, embeddings: np.ndarray) -> faiss.IndexFlatIP:
        dimension = embeddings.shape[1]  # Number of features in each embedding
        logger.info(f"Building index with model:{self.embedding_model} with dimension: {dimension}")
        # Create a FAISS index
        self.document_index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity search
        # Add embeddings to the index
        self.document_index.add(embeddings)
        return self.document_index
    
    def as_retriever(self, query_text, top_k=appConfig["top_k"]):
        return self.search(query_text, top_k)
    

    def get_embeddings(self, documents) -> np.ndarray:
        embeddings = []
        for order, doc in enumerate(documents, start=1):
            logger.info(f"Generating embedding for document id: {doc[0]}")
            response = ollama.embeddings(model=self.embedding_model, prompt=doc[1])
            embeddings.append(response["embedding"])
            embedding_vector = np.array(response["embedding"], dtype="float32")
            self.cursor.execute(
                """INSERT INTO document_embeddings (faiss_index_id, document_page_split_id, embedding) 
                VALUES (?, ?, ?)""",
                (order, doc[0], embedding_vector.tobytes()),
            )
        self.conn.commit()
        # Convert embeddings to a NumPy array
        embeddings = np.array(embeddings).astype("float32")
        return embeddings

    @staticmethod
    def get_embedding(text, model=appConfig["EMBEDDING_MODEL"]) -> np.ndarray:
        logger.info(f"Generating embedding for text: {text}")
        response = ollama.embeddings(model=model, prompt=text)
        embeddings = []
        embeddings.append(response["embedding"])
        embeddings = np.array(embeddings).astype("float32")
        return embeddings


    @staticmethod
    def get_doc(faiss_index, config=appConfig):
        logger.info("Loading documents from database for faiss_index {}".format(faiss_index))
        conn = sqlite3.connect(config["DB_NAME"])
        cursor = conn.cursor()
        cursor.execute(
            f"""SELECT text from document_page_split 
                WHERE id IN ( 
                SELECT document_page_split_id 
                FROM document_embeddings 
                WHERE faiss_index_id = {faiss_index})"""
            )
        document = cursor.fetchone()
        if document is None:
            logger.warn(f"Document not found for faiss_index {faiss_index}")
            return ""
        return document[0]


    @staticmethod
    def save(index, filename):
        logger.info("Saving index to file")
        faiss.write_index(index, filename)
        logger.info("Index saved successfully")

    @staticmethod
    def load(filename):
        logger.info("Loading index from file")
        loaded_index = faiss.read_index(filename)
        logger.info("Index loaded successfully")
        return loaded_index

    @staticmethod
    def load_documents_from_db(config, max_documents: int = 1000) -> list:
        """Load documents from the downloads database table."""
        conn = sqlite3.connect(config["DB_NAME"])
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT id, text from document_page_split LIMIT {max_documents}"
        )
        documents = cursor.fetchall()
        conn.close()
        return [(doc[0], FaissDB.clean_text(doc[1])) for doc in documents]

    @staticmethod
    def clean_text(md_text):
        """Remove Markdown syntax and extract clean text."""
        md_text = re.sub(r"\[.*?\]\(.*?\)", "", md_text)  # Remove links
        md_text = re.sub(r"#{1,6}\s*", "", md_text)  # Remove headers
        md_text = re.sub(
            r"(```.*?```|`.*?`)", "", md_text, flags=re.DOTALL
        )  # Remove code blocks
        md_text = re.sub(
            r"\*{1,2}|\_{1,2}", "", md_text
        )  # Remove bold/italic formatting
        md_text = re.sub(r">\s*", "", md_text)  # Remove block quotes
        md_text = re.sub(r"[-+*]\s+", "", md_text)  # Remove bullet points
        md_text = re.sub(r"\d+\.\s+", "", md_text)  # Remove numbered lists
        return md_text.strip()

```

Using the class

```python

db = Database()
vector_store = FaissDB()
docs = vector_store.load_documents_from_db(appConfig, 30)
embeddings = vector_store.get_embeddings(docs)
index = vector_store.build_index(embeddings)
vector_store.set_index(index)

# Query embedding (generate an embedding for the query)
query_text = "RAG Database"
query_embedding = vector_store.get_embedding(query_text)

# Search for the top-k most similar embeddings
k = 5  # Number of nearest neighbors to retrieve
distances, indices = index.search(query_embedding, k)

# Print results
print("Indices of similar documents:", indices)
print("Distances to similar documents:", distances)
print("Similar documents:")
for i, idx in enumerate(indices[0]):
    print(f"Document {idx}: {docs[idx][1]}")
    doc = FaissDB.get_doc(idx)
    print("---------------------")
    print(doc)
    print("---------------------")
    print("Similarity score:", 1 - distances[0][i])
```
