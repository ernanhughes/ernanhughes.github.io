+++
date = '2025-02-07T19:33:51Z'
draft = false
title = 'DeepResearch Part 2: Building a RAG Tool for arXiv PDFs'
categories = ['arXiv', 'RAG', 'Vector Database', 'Ollama']
tags = ['arxiv', 'rag','ollama', 'agent']
+++


### Summary

In this post, we'll build a Retrieval Augmented Generation (RAG) tool to process the PDF files downloaded from arXiv in the previous post [DeepResearch Part 1]({{< relref "post/deepresearch1.md" >}}). 
This RAG tool will be capable of loading, processing, and semantically searching the document content. 
It's a versatile tool applicable to various text sources, including web pages.

### Building the RAG Tool

Following up on our arXiv downloader, we now need a tool to process the downloaded PDFs.  This post details the creation of such a tool.


### Enhanced Configuration (`Config`)

Some extra config options

```python

class Config:
    DB_NAME = "pdf_documents.db"
    TABLE_NAME = "pdf_documents"
    PAGES_TABLE = "pdf_pages"
    FILES_TABLE = "pdf_files"
    LOOKUP_TABLE = "pdf_lookup"
    EMBEDDING_MODEL_NAME = "mxbai-embed-large"   # Or another Ollama-compatible model
    OLLAMA_BASE_URL = "http://localhost:11434"
    VECTOR_DB = "pdf_vector.db"
    DIMS = 1024  # Define vector dimensions

    @classmethod
    def load_from_file(cls, config_file="config.json"):
        if os.path.exists(config_file):
            try:
                with open(config_file, "r") as f:
                    config_data = json.load(f)
                for key, value in config_data.items():
                    if hasattr(cls, key):
                        setattr(cls, key, value)
                logging.info("Loaded configuration from file.")
            except Exception as e:
                logging.error(f"Failed to load configuration file: {e}")

# Load configuration from file if available
Config.load_from_file()

```

### Database Class (`PDFDatabase`)

A dedicated database class helps manage our PDF data.

```python
import sqlite3
from pypdf import PdfReader

class PDFDatabase:
    def __init__(self, db_name=Config.DB_NAME):
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self.create_tables()
    
    def create_tables(self):
        self.cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {Config.TABLE_NAME} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT UNIQUE,
                text TEXT
            )
        ''')
        
        self.cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {Config.PAGES_TABLE} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER,
                page_number INTEGER,
                text TEXT,
                FOREIGN KEY(document_id) REFERENCES {Config.TABLE_NAME}(id)
            )
        ''')
        
        self.cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {Config.FILES_TABLE} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT UNIQUE,
                file BLOB
            )
        ''')
        
        self.cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {Config.LOOKUP_TABLE} (
                id INTEGER PRIMARY KEY,
                content TEXT
            )
        ''')
        self.conn.commit()
    
    def close(self):
        self.conn.close()
```

Now insert the files

```python
# Insert PDFs into the database
db = PDFDatabase()
for file_path in pdf_files:
    db.insert_pdf(file_path)
db.close()

print("PDF files, content, and pages inserted into the database successfully.")
```

### Embedding Generation

We'll use Ollama to generate embeddings.

```python
import requests
import json



def generate_embeddings(text, model_name=Config.EMBEDDING_MODEL_NAME):
    try:
        url = f"{Config.OLLAMA_BASE_URL}/api/embed"
        data = {"input": text, "model": model_name}
        response = requests.post(url, json=data)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        embeddings = response.json()
        return embeddings["embeddings"]
    except requests.exceptions.RequestException as e: # Catch connection and HTTP errors
        logging.error(f"Failed to generate embeddings: {e}")
        return None
    except (json.JSONDecodeError, KeyError) as e: # Catch JSON errors
        logging.error(f"Failed to parse JSON response: {e}")
        return None
```

### Using sqlite as a vector store

I will be using [sqlite_vec](https://github.com/asg017/sqlite-vec) as my vector store. I chose this for simplicity. It will just work with sqlite anywhere.



These are some helper functions.

```python
import numpy as np

# Function to serialize float32 list to binary format compatible with sqlite-vec
def serialize_f32(vec):
    return np.array(vec, dtype=np.float32).tobytes()

def reciprocal_rank_fusion(fts_results, vec_results, k=60):
    rank_dict = {}

    # Process FTS results
    for rank, (id,) in enumerate(fts_results):
        if id not in rank_dict:
            rank_dict[id] = 0
        rank_dict[id] += 1 / (k + rank + 1)

    # Process vector results
    for rank, (rowid, distance) in enumerate(vec_results):
        if rowid not in rank_dict:
            rank_dict[rowid] = 0
        rank_dict[rowid] += 1 / (k + rank + 1)

    # Sort by RRF score
    sorted_results = sorted(rank_dict.items(), key=lambda x: x[1], reverse=True)
    return sorted_results

def or_words(input_string):
    # Split the input string into words
    words = input_string.split()

    # Join the words with ' OR ' in between
    result = ' OR '.join(words)

    return result

def lookup_row(id):
    row_lookup = cur.execute('''
    SELECT content FROM pdf_lookup WHERE id = ?
    ''', (id,)).fetchall()
    content = ''
    for row in row_lookup:
        content= row[0]
        break
    return content
```

#### **1. `serialize_f32(vec)`**
This function converts a list of `float32` values into a binary format that is compatible with `sqlite-vec` (an extension for handling vector data in SQLite).

- `np.array(vec, dtype=np.float32)`: Converts the input list into a NumPy array of type `float32`.
- `.tobytes()`: Serializes the array into bytes, which can be stored in SQLite as a BLOB.

---

#### **2. `reciprocal_rank_fusion(fts_results, vec_results, k=60)`**
This function performs **Reciprocal Rank Fusion (RRF)**, a method used to merge multiple ranked search results. It takes:
- `fts_results`: A list of (id,) tuples from full-text search (FTS).
- `vec_results`: A list of (rowid, distance) tuples from vector search.
- `k`: A hyperparameter that controls ranking fusion.

#### **How it works:**
1. It initializes an empty dictionary `rank_dict` to store ranking scores.
2. It processes `fts_results`, giving each ID a score based on the formula:  
   \[
   \frac{1}{k + \text{rank} + 1}
   \]
   - This ensures higher-ranked results get more weight.
3. It does the same for `vec_results`, using `rowid` and `distance` (although `distance` is not directly used here).
4. The dictionary is then sorted by the computed scores, returning a list of IDs sorted by rank fusion.

**Why use RRF?**  
- It balances different ranking methods (e.g., textual search vs. semantic vector search).
- Helps when individual ranking methods have different scoring systems.

---

#### **3. `or_words(input_string)`**
This function converts a given input string into a format suitable for an **OR-based** full-text search query.

#### **How it works:**
1. Splits the input string into individual words.
2. Joins them using `' OR '` to construct an FTS search query.

**Example Usage:**
```python
or_words("hello world")
# Output: "hello OR world"
```
This means that a full-text search will return results containing **either** "hello" **or** "world".

---

#### **4. `lookup_row(id)`**
This function retrieves the content of a document based on its `id` from the `pdf_lookup` table in SQLite.

#### **How it works:**
1. Executes an SQL `SELECT` query to fetch the `content` where `id = ?`.
2. Uses `.fetchall()` to get all matching rows.
3. Iterates over the results and assigns the first row's content to the `content` variable.
4. Returns the `content`.

**Potential Issues:**
- `cur` (the database cursor) is not defined in this code snippet, so it assumes there's a previously established SQLite connection.
- If multiple rows exist for the same `id`, only the first row's content is returned.

---




### Enabling sqlite_vec

```python
import sqlite_vec

# Create an in memory sqlite db
db = sqlite3.connect("pdf_vector.db")
db.enable_load_extension(True)
sqlite_vec.load(db)
db.enable_load_extension(False)

sqlite_version, vec_version = db.execute(
    "select sqlite_version(), vec_version()"
).fetchone()
print(f"sqlite_version={sqlite_version}, vec_version={vec_version}")

```

```
sqlite_version=3.45.1, vec_version=v0.1.6
```

Lets just test our embeddings generation

```python
data = generate_embeddings('The quick brown fox')
dims = len(data[0])
print ('Dims in Vector Embeddings:', dims)
```

We see out model has 1024 dimensions.

```
Dims in Vector Embeddings: 1024
```


### Creating our vector store

```python
cur = db.cursor()
cur.execute('CREATE VIRTUAL TABLE IF NOT EXISTS pdf_fts USING fts5(id UNINDEXED, content, tokenize="porter unicode61");')

# sqlite-vec always adds an ID field
cur.execute('''CREATE VIRTUAL TABLE IF NOT EXISTS pdf_vec USING vec0(embedding float[''' + str(dims) + '])''')

# Create a content lookup table with an index on the ID
cur.execute('CREATE TABLE IF NOT EXISTS pdf_lookup (id INTEGER PRIMARY KEY AUTOINCREMENT, content TEXT);')

```


### Testing our vector store

```python 
fts_data = [
    (1, 'The quick brown fox jumps over the lazy dog.'),
    (2, 'Artificial intelligence is transforming the world.'),
    (3, 'Climate change is a pressing global issue.'),
    (4, 'The stock market fluctuates based on various factors.'),
    (5, 'Remote work has become more prevalent during the pandemic.'),
    (6, 'Electric vehicles are becoming more popular.'),
    (7, 'Quantum computing has the potential to revolutionize technology.'),
    (8, 'Healthcare innovation is critical for societal well-being.'),
    (9, 'Space exploration expands our understanding of the universe.'),
    (10, 'Cybersecurity threats are evolving and becoming more sophisticated.')
]

cur.execute("DELETE FROM pdf_fts")
cur.execute("DELETE FROM pdf_lookup")
cur.execute("DELETE FROM pdf_vec")


cur.executemany('''
INSERT INTO pdf_fts (id, content) VALUES (?, ?)
''', fts_data);


cur.executemany('''
  INSERT INTO pdf_lookup (id, content) VALUES (?, ?)
''', fts_data)


# Generate embeddings for the content and insert into mango_vec
for row in fts_data:
    id, content = row
    embedding = generate_embeddings(content)
    cur.execute('''
    INSERT INTO pdf_vec (rowid, embedding) VALUES (?, ?)
    ''', (id, serialize_f32(list(embedding))))


def search(query: str = "Electric", top_k: int = 2):
    fts_results = cur.execute('''
    SELECT id FROM pdf_fts WHERE pdf_fts MATCH ? ORDER BY rank limit 5
    ''', (or_words(query),)).fetchall()

    # Vector search query
    query_embedding = generate_embeddings(query)
    vec_results = cur.execute('''
    SELECT rowid, distance FROM pdf_vec WHERE embedding MATCH ? and K = ?
    ORDER BY distance
    ''', [serialize_f32(list(query_embedding)), top_k]).fetchall()

    # Combine results using RRF
    combined_results = reciprocal_rank_fusion(fts_results, vec_results)

    # Print combined results
    for id, score in combined_results:
        print(f'ID: {id}, Content: {lookup_row(id)}, RRF Score: {score}')    


print("---- technology ----")
search("technology")
print("---- Electric ----")
search("Electric")  
print("---- Medical ----")
search("medical")

```


```
---- technology ----
ID: 7, Content: Quantum computing has the potential to revolutionize technology., RRF Score: 0.03278688524590164
ID: 2, Content: Artificial intelligence is transforming the world., RRF Score: 0.016129032258064516
---- Electric ----
ID: 6, Content: Electric vehicles are becoming more popular., RRF Score: 0.03278688524590164
ID: 2, Content: Artificial intelligence is transforming the world., RRF Score: 0.016129032258064516
---- Medical ----
ID: 8, Content: Healthcare innovation is critical for societal well-being., RRF Score: 0.01639344262295082
ID: 3, Content: Climate change is a pressing global issue., RRF Score: 0.016129032258064516

```

### Vector Database (`VectorDB`)

We encapsulate the vector database operations within a `VectorDB` class.

```python
class VectorDB:
    def __init__(self, db_name=Config.VECTOR_DB):
        self.conn = sqlite3.connect(db_name)
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)
        self.cursor = self.conn.cursor()
        self.create_tables()
        self.dims = Config.DIMS 
    
    def create_tables(self):
        self.cursor.execute(f'CREATE VIRTUAL TABLE IF NOT EXISTS pdf_vec USING vec0(embedding float[{self.dims}])')
        self.cursor.execute('CREATE TABLE IF NOT EXISTS pdf_lookup (id INTEGER PRIMARY KEY AUTOINCREMENT, content TEXT);')
        self.conn.commit()
    
    def search(self, query: str, top_k: int = 2):
        fts_results = self.cursor.execute("SELECT id FROM pdf_fts WHERE pdf_fts MATCH ? ORDER BY rank LIMIT 5", (query,)).fetchall()

        query_embedding = EmbeddingModel.generate_embeddings(query)
        if query_embedding is None:
            logging.error(f"Failed to generate embedding for search query: {query}.")
            return []
        vec_results = self.cursor.execute(f"SELECT rowid, distance FROM pdf_vec WHERE embedding MATCH ? AND K = ? ORDER BY distance", [np.array(query_embedding, dtype=np.float32).tobytes(), top_k]).fetchall()
        
        combined_results = VectorDB.reciprocal_rank_fusion(fts_results, vec_results)
        for id, score in combined_results:
            print(f'ID: {id}, Content: {self.lookup_row(id)}, RRF Score: {score}')

    @staticmethod
    def serialize_f32(vec):
        return np.array(vec, dtype=np.float32).tobytes()

    @staticmethod
    def reciprocal_rank_fusion(fts_results, vec_results, k=60):
        rank_dict = {}
        # Process FTS results
        for rank, (id,) in enumerate(fts_results):
            if id not in rank_dict:
                rank_dict[id] = 0
            rank_dict[id] += 1 / (k + rank + 1)

        # Process vector results
        for rank, (rowid, distance) in enumerate(vec_results):
            if rowid not in rank_dict:
                rank_dict[rowid] = 0
            rank_dict[rowid] += 1 / (k + rank + 1)

        # Sort by RRF score
        sorted_results = sorted(rank_dict.items(), key=lambda x: x[1], reverse=True)
        return sorted_results

    @staticmethod
    def or_words(input_string):
        # Split the input string into words
        words = input_string.split()
        # Join the words with ' OR ' in between
        result = ' OR '.join(words)
        return result

    def lookup_row(self, id):
        row_lookup = self.cursor.execute('''
        SELECT content FROM pdf_lookup WHERE id = ?
        ''', (id,)).fetchall()
        content = ''
        for row in row_lookup:
            content= row[0]
            break
        return content


    def close(self):
        self.conn.close()

class EmbeddingModel:
    @staticmethod
    def generate_embeddings(text, model_name=Config.EMBEDDING_MODEL_NAME):
        try:
            print(f"Model: {model_name}")
            url = f"{Config.OLLAMA_BASE_URL}/api/embed"
            response = requests.post(url, json={"input": text, "model": model_name})
            response.raise_for_status()
            return response.json()["embeddings"]
        except requests.RequestException as e:
            logging.error(f"Embedding generation failed: {e}")
            return None

```


### Create the tool to use the Vector database


```python

from smolagents import Tool

class RetrieverTool(Tool):
    name = "retriever"
    description = (
        "Uses semantic search to retrieve the parts of documentation that could be most relevant to answer your query."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question.",
        }
    }
    output_type = "string"

    def __init__(self, vector_store, **kwargs):
        super().__init__(**kwargs)
        self.vector_store = vector_store

    def forward(self, query: str) -> str:
        results = self.vector_store.search(query, top_k=3) 
        if not results:
            return "No relevant documents found."

        formatted_results = "\nRetrieved documents:\n"
        for i, (id, score) in enumerate(results):
            content = self.vector_store.lookup_row(id)
            formatted_results += f"\n\n===== Document {i+1} (Score: {score:.4f}) =====\n{content}"

        return formatted_results

vector_store = VectorDB()
retriever_tool = RetrieverTool(vector_store=vector_store)

```

### Using the tool

```python

from smolagents import LiteLLMModel, CodeAgent

model = LiteLLMModel(
    model_id="ollama_chat/qwen2.5",
    api_base="http://localhost:11434/",
)

agent = CodeAgent(
    tools=[retriever_tool],
    model=model,
    max_steps=4,
    verbosity_level=2,
)

agent_output = agent.run("What are some advanced applications of Cellular Automata?")

```
