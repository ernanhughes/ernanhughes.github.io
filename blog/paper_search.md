+++
date = '2025-02-22T07:28:15Z'
draft = false
title = 'Automating Paper Retrieval and Processing with PaperSearch'
categories = ['Ollama', 'SmartAnswer', 'PaperSearch']
tags = ['Ollama', 'SmartAnswer', 'PaperSearch']
+++

## Summary

This is part on in a series of blog post working towards `SmartAnswer` a comprehensive improvement to how `Large Language Models` LLMs answer questions.

This tool will be the source of data for `SmartAnswer` and allow it to find and research better data when generating answers.

I want this tool to be included in that solution but I dot want all the code from this tool distracting from the `SmartAnswer` solution. Hence this post.

## Why do we need Paper Search
I think papers provide a rich source of advanced new quality data. The web, youtube, github are amazing however I believe the quality and uniqueness of content to data is lower than for papers.

This tool automates the retrieval of research papers from [**arXiv**](https://arxiv.org/), extracting their content, chunking the text for easier processing, and generating embeddings for further analysis.

This tool can be the source of extra data in our projects or the source of data in tour projects. It could also be used to compliment or enhance data in our projects.

## Why PaperSearch?

Academic papers represent the forefront of human knowledge, offering rigorously peer-reviewed, methodologically sound, and formally written content. However, extracting and processing this knowledge is challenging due to the unstructured nature of PDFs and the lack of tools for efficient retrieval and analysis.

PaperSearch addresses these challenges by automating the retrieval, extraction, and structuring of research papers. It enables users to:

* **Search** for relevant papers using arXiv’s API.
* **Download** and store PDFs in a structured database.
* **Extract** text while preserving semantic integrity.
* **Chunk** text for efficient processing.
* **Generate embeddings** for semantic search and retrieval.

---

## How PaperSearch Works

### 1. Searching and Retrieving Papers

The `search` method queries the **arXiv API**, retrieving relevant papers based on a search query. It extracts PDF links and downloads them if they don’t already exist in the database.

```python
params = {"search_query": query, "start": 0, "max_results": self.max_results}
response = requests.get(self.base_url, params=params)
```

Once retrieved, the PDFs are saved locally and recorded in a **SQLite database**.

---

### 2. Extracting Text from PDFs

After downloading, **PaperSearch** extracts the textual content using **PyPDF2**:

```python
reader = PyPDF2.PdfReader(file)
text = page.extract_text()
```

Each page’s content is stored in the database for efficient retrieval and further processing.

---

### 3. Chunking Text with Sentence Awareness

To optimize **retrieval-based tasks**, the text is split into manageable chunks using **NLTK’s sentence tokenizer**, ensuring that chunks do not break in the middle of sentences:

```python
sentences = sent_tokenize(text)
if len(current_chunk) + len(sentence) <= self.chunk_size:
    current_chunk += " " + sentence
else:
    chunks.append(current_chunk.strip())
    current_chunk = sentence
```

This structured approach allows for effective processing in downstream tasks such as **semantic search** and **language model input processing**.


---

### 4. Generating Embeddings for Semantic Search

A key feature of **PaperSearch** is its ability to generate **embeddings** for extracted text, enabling similarity-based retrieval. It leverages an **embedding model** to encode document content:

```python
embeddings = generate_embeddings(text)
self.cursor.execute(
    "UPDATE document_text SET embeddings = ? WHERE id = ?",
    (embeddings, document_text_id),
)
```

These embeddings facilitate **vector search**, making it easier to find related documents and improve RAG workflows.

---

## The Bigger Picture

**PaperSearch** is a component of an **AI-powered research assistant**. By integrating it with **vector databases, LLMs, and knowledge graphs**, users can build powerful applications for research automation, question answering, and literature analysis.

### Future Enhancements

Some planned improvements include:

- **HTML download** A lot of papers now have HTML version available online this will allow for better processing of the paper.
- **Graph generation** Each paper is built from a larger collection of papers. It will build a graph of related papers using these links this and a configurable dept of retrieval. It will then retrieve all related papers.
- **Multi-source support** (e.g., Semantic Scholar, Google Scholar API integration)
- **Metadata extraction** for improved paper classification
- **Interactive UI for paper search and retrieval**
- **Related Research Detection** It will support the analysis and detection of new research. Say you are researching something  and have a paper in progress if will scan the latest papers for similar new papers and extract the useful or similar research form those papers.
- **Error Detection** it will use the latest papers to determine the quality of it current or a selected paper. 
- **Web Search** similar tool to get data from the web/s3... any remote system.
- **File Search** I have a large amount of local code and data that is not public. I want a similar tool for processing that data.

With tools like **PaperSearch**, researchers and developers can focus on analysis rather than manual information retrieval, enhancing productivity in AI and academic research.

---

### Get Started with PaperSearch

In the next few poets I will be building on this tool towards a solution.

---

## Current code

The idea behind this blog is to get to the edge of AI research. In the future the AI will be writing the code. 
Until then we will be writing it aided by the AI. 
That is why I include a lot of code in the posts.  
In my opinion the future code will be a pseudocode of a prompt with an intent that a model can use to generate a solution. Think of a small scrum task. That is for another blog post later.


We need to create a database to hold data related to the papers we find.

1. We store the full paper in the database.
2. We store the fully extracted text in there
3. We store each page with its data.
4. We chunk the text currently that is stored also. These chunks will become our documents in our RAG retrieval approach.
5. We have a mapping between the Faiss index and the chunk in our extraction.
6. We store the generated embedding. 

```sql
CREATE TABLE IF NOT EXISTS query (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT UNIQUE,
    datetime TEXT
);

CREATE TABLE IF NOT EXISTS paper_search (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT,
    pdf_url TEXT,
    filename TEXT,
    pdf_data BLOB
);

CREATE TABLE IF NOT EXISTS document_page (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_search_id INTEGER,
    page_number INTEGER,
    text TEXT,
    FOREIGN KEY (paper_search_id) REFERENCES paper_search(id)
);

CREATE TABLE IF NOT EXISTS document_text (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_search_id INTEGER,
    text TEXT,
    FOREIGN KEY (paper_search_id) REFERENCES paper_search(id)
);

CREATE TABLE IF NOT EXISTS document_page_split (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_page_id INTEGER,
    text TEXT,
    FOREIGN KEY (document_page_id) REFERENCES document_page(id)
);

CREATE TABLE IF NOT EXISTS document_index (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_index INTEGER,
    document_page_id INTEGER,
    text TEXT,
    FOREIGN KEY (document_page_id) REFERENCES document_page(id)
);

CREATE TABLE IF NOT EXISTS document_embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    faiss_index_id INTEGER,
    document_page_split_id INTEGER,
    embedding BLOB,
    FOREIGN KEY (document_page_split_id) REFERENCES document_page_split(id)
);

```

Database class for managing creation of the database.

```python 
import sqlite3

from config import appConfig
import logging

logger = logging.getLogger(__name__)


class Database:
    def __init__(self, db_name=appConfig["DB_NAME"], schema_file=appConfig["SCHEMA_FILE"]):
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self.load_schema(schema_file)

    def load_schema(self, schema_file):
        """Load the database schema from an external SQL file."""
        logger.debug(f"Loading schema from {schema_file}")
        with open(schema_file, "r") as f:
            schema_sql = f.read()
            logger.debug(f"Loaded schema from {schema_file}")
        logger.debug(f"Executing schema SQL: {schema_sql}")
        self.cursor.executescript(schema_sql)
        self.conn.commit()
        logger.debug("Schema loaded successfully")

    def close(self):
        """Close the database connection."""
        self.conn.close()
```

```python

import requests
import xml.etree.ElementTree as ET
import os
import logging
import PyPDF2
import datetime

from config import appConfig
from ollama_utils import generate_embeddings
from nltk.tokenize import sent_tokenize

logger = logging.getLogger(__name__)

from database  import Database

class PaperSearch:
    def __init__(self, config=appConfig):
        self.config = config
        self.base_url = "http://export.arxiv.org/api/query"
        self.max_results = config["max_search_results"]
        self.data_dir = config["DATA_DIR"]
        self.chunk_size = config["CHUNK_SIZE"]
        self.chunk_overlap = config["CHUNK_OVERLAP"]
        self.db = Database(config["DB_NAME"])
        self.conn = self.db.conn
        self.cursor = self.db.cursor


    def search(self, query):
        """Search Arxiv and download related PDFs, storing results in the database."""
        params = {"search_query": query, "start": 0, "max_results": self.max_results}
        response = requests.get(self.base_url, params=params)
        if response.status_code == 200:
            pdf_links = self.get_pdf_links(response)
            downloaded_pdfs = []
            if not os.path.exists(self.data_dir):
                os.makedirs(self.data_dir)
            for pdf_url in pdf_links:
                pdf_file = self.download_pdf(pdf_url, query)
                if pdf_file:
                    downloaded_pdfs.append(pdf_file)
            logger.info(f"Downloaded {len(downloaded_pdfs)} PDFs for query: {query}")
            return downloaded_pdfs
        return []

    def get_pdf_links(self, response):
        """Extract PDF links from an Arxiv API response."""
        root = ET.fromstring(response.text)
        pdf_links = []
        for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
            for link in entry.findall("{http://www.w3.org/2005/Atom}link"):
                if link.attrib.get("title") == "pdf":
                    pdf_links.append(link.attrib.get("href"))
        return pdf_links

    def download_pdf(self, pdf_url, query):
        """Download a PDF file from a URL and store it in the database."""
        # Check if the document already exists in the database
        self.cursor.execute("SELECT id FROM paper_search WHERE filename = ?", (pdf_url,))
        result = self.cursor.fetchone()
        if result:
            logger.info(f"PDF already exists in database: {pdf_url}")
            return None  
        filename = os.path.join(self.data_dir, pdf_url.split("/")[-1])
        pdf_response = requests.get(pdf_url)
        if pdf_response.status_code == 200:
            pdf_data = pdf_response.content
            with open(filename, "wb") as f:
                f.write(pdf_data)
            self.cursor.execute(
                "INSERT INTO paper_search (query, pdf_url, filename, pdf_data) VALUES (?, ?, ?, ?)",
                (query, pdf_url, filename, pdf_data),
            )
            self.conn.commit()
            self.extract_text_from_pdf(filename)
            logger.info(f"Downloaded PDF: {pdf_url}")
            return filename
        return None
    
    def insert_query(self, query):
        self.cursor.execute('''INSERT OR IGNORE INTO query (query, datetime) VALUES (?, ?)''',
                       (query, datetime.now().isoformat()))
        self.conn.commit()
        self.cursor.execute('''SELECT id FROM query WHERE query = ?''', (query,))
        return self.cursor.fetchone()[0]


    def extract_text_from_pdf(self, pdf_file):
        """Extract text from a PDF file and store them in the database."""
        extracted_text = []
        with open(pdf_file, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            paper_search_id = self.cursor.execute("SELECT id FROM paper_search WHERE filename = ?", (pdf_file,)).fetchone()[0]
            for page in reader.pages:
                text = page.extract_text()
                extracted_text.append(text)
                self.cursor.execute(
                    """INSERT INTO document_page (paper_search_id, page_number, text) 
                    VALUES (?, ?, ?)
                    RETURNING id""",
                    (paper_search_id, pdf_file, text),
                )
                page_id = self.cursor.lastrowid
                self.split_text_on_sentences(page_id, text)
            self.conn.commit()

            extracted_text.append(text)
            self.cursor.execute(
                "INSERT INTO document_text (paper_search_id, text) VALUES (?, ?)",
                (paper_search_id, text),
            )
        self.conn.commit()
        return extracted_text

    def split_text_on_sentences(self, page_id, text):
        """Split text into chunks while ensuring sentence boundaries and store in database."""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += " " + sentence
            else:
                chunk_text = current_chunk.strip()
                chunks.append(chunk_text)
                current_chunk = sentence
                print(chunk_text)
        if current_chunk:
            chunks.append(current_chunk.strip())
        logger.info(f"Inserting {len(chunks)} chunks for text length {len(text)}")
        self.cursor.executemany(
            "INSERT INTO document_page_split(document_page_id, text) VALUES (?, ?)",
            [(page_id, chunk_text) for chunk_text in chunks]
        )
        logger.info(f"Inserted {len(chunks)} chunks")
        self.conn.commit()
        return chunks

    def create_embeddings(self):
        """Create embeddings for the extracted text."""
        self.cursor.execute("SELECT id, text FROM document_text")
        rows = self.cursor.fetchall()
        for row in rows:
            document_text_id, text = row
            embeddings = generate_embeddings(text)
            self.cursor.execute(
                "UPDATE document_text SET embeddings = ? WHERE id = ?",
                (embeddings, document_text_id),
            )
        self.conn.commit()

```

Using the tool

```python

ps = PaperSearch(appConfig)
ps.search("RAG Fusion")

```