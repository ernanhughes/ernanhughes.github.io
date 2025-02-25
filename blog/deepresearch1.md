+++
date = '2025-02-07T08:33:23Z'
draft = false
title = 'DeepResearch Part 1: Building an arXiv Search Tool with SmolAgents'
categories = ['arXiv', 'SmolAgents', 'Python']
tags = ['arxiv', 'smolagents', 'agent']

+++

### Summary

This post kicks off a series of three where we'll build, extend, and use the open-source DeepResearch application inspired by the [Hugging Face blog post](https://huggingface.co/blog/open-deep-research).  In this first part, we'll focus on creating an arXiv search tool that can be used with SmolAgents.


DeepResearch aims to empower research by providing tools that automate and streamline the process of discovering and managing academic papers. This series will demonstrate how to build such tools, starting with a powerful arXiv search tool.


### The arXiv Search Tool

Our tool will automate the following tasks:

* **arXiv Search:** Query the arXiv API for relevant papers.
* **Paper Download:** Download the PDF files of the found papers.
* **Metadata Storage:** Store paper metadata (title, link, file path, hash) and the PDF content in a SQLite database.
* **Logging:** Maintain comprehensive logs of all actions and errors in a file and a database.
* **Configuration:** Manage settings (search query, output folder, API base URL, log database path) using a flexible configuration system.


### Configuration Class (`Config`)

A dedicated configuration class makes managing settings easy and organized.

The can be a game changer if you want to build multiple pipelines of research tasks.

```python
# Configuration Class
import os

class Config:
    def __init__(self):
        self.search_query = "agent"
        self.max_results = 50
        self.output_folder = "data"
        self.base_url = "http://export.arxiv.org/api/query?"
        self.log_db_path = "app_logs.db"

    def load_from_env(self):
        self.search_query = os.environ.get("SEARCH_QUERY", self.search_query)
        self.max_results = int(os.environ.get("MAX_RESULTS", self.max_results))
        self.output_folder = os.environ.get("OUTPUT_FOLDER", self.output_folder)
        self.base_url = os.environ.get("BASE_URL", self.base_url)
        self.log_db_path = os.environ.get("LOG_DB_PATH", self.log_db_path)
        return self

    def load_from_file(self, file_path):
        """Loads configuration from a file, expecting key=value pairs."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Config file '{file_path}' not found.")

        with open(file_path, "r") as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith("#"):  # Ignore empty lines and comments
                    continue
                key, value = map(str.strip, line.split("=", 1))
                if hasattr(self, key):  # Only update existing attributes
                    try:
                        if isinstance(getattr(self, key), int):
                            value = int(value)
                        setattr(self, key, value)
                    except ValueError:
                        raise ValueError(f"Invalid type for key: {key}, expected {type(getattr(self, key)).__name__}")
        return self


```

To use the config we would create a configuration file:  `config.txt`.

```ini, TOML
SEARCH_QUERY=deep learning
MAX_RESULTS=100
OUTPUT_FOLDER=results
BASE_URL=http://export.arxiv.org/api/query?
LOG_DB_PATH=logs.db
```

Load the file

```python
config = Config().load_from_file("config.txt")
print(config.search_query)  # Output: deep learning
print(config.max_results)   # Output: 100
```

### Logging 

We'll implement logging to a file, the console, and a database for comprehensive tracking.

```python

import sqlite3
import datetime
import logging
import os

class DatabaseHandler(logging.Handler):
    def __init__(self, db_path):
        super().__init__()
        self.db_path = db_path
        self._initialize_db()

    def _initialize_db(self):
        """Ensures the logs table exists in SQLite."""
        try:
            with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        level TEXT NOT NULL,
                        message TEXT NOT NULL
                    )
                """)  # Executes the statement separately
                conn.commit()
        except sqlite3.Error as e:
            print(f"Error initializing database: {e}")

    def emit(self, record):
        """Inserts a log record into the database safely."""
        try:
            with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
                cursor = conn.cursor()
                timestamp = datetime.datetime.fromtimestamp(record.created).isoformat()
                log_entry = (timestamp, record.levelname, record.getMessage())
                cursor.execute(
                    "INSERT INTO logs (timestamp, level, message) VALUES (?, ?, ?)",
                    log_entry
                )  # Ensuring a single statement is executed at a time
                conn.commit()
        except sqlite3.Error as e:
            print(f"Database error: {e}")
        except Exception as e:
            print(f"Unexpected error in DatabaseHandler: {e}")


# Ensure Config is initialized before logging
config = Config().load_from_env()

# Ensure log directory exists
if not os.path.exists(config.output_folder):
    os.makedirs(config.output_folder)

# Set up logging
log_file_path = os.path.join(config.output_folder, "app.log")

file_handler = logging.FileHandler(log_file_path)
db_handler = DatabaseHandler(config.log_db_path)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
file_handler.setFormatter(formatter)
db_handler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logger.addHandler(file_handler)
logger.addHandler(db_handler)

```

There is drawbacks here. If the database gets locked it will stop logging. You can only access the database connection on the thread it was created on.


### ArXiv search tool (`fetch_arxiv_papers`)

This tool is designed to be used with SmolAgents via the @tool decorator.

This tool is a fully automated arXiv research paper downloader and database manager. It allows users to search, fetch, and store research papers from arXiv based on a specified query. Here’s how it works:

* **Search Automation** – The tool queries arXiv for research papers related to a given topic.
* **Metadata Extraction** – It parses the arXiv API response to extract paper titles and PDF links.
* **Smart Downloading** – It checks if the file already exists before downloading, preventing duplicate downloads.
* **Secure Storage** – Each paper is stored in a SQLite database, including metadata, file hash, and the actual PDF content.
* **Logging & Error Handling** – Comprehensive logging ensures that failures are tracked while allowing the tool to continue processing other papers.
* **Performance Optimizations** – Efficient file hashing and database lookups reduce redundant operations.

This tool will be a component in our deep research application.

```python

from smolagents import tool
import xml.etree.ElementTree as ET


from typing import Optional, List, Tuple
import os
import time
import requests
from pathlib import Path
from urllib.parse import urlparse
import hashlib
import re
import sqlite3

@tool
def fetch_arxiv_papers(search_query: str = None, max_results: Optional[int] = None) -> List[Tuple[str, str]]:
    """
    Searches arXiv for research papers on a topic and saves the papers to a folder
    Args:
        search_query: the topic to search for
        max_results: max results to return
    """

    search_query = search_query or config.search_query
    max_results = max_results or config.max_results

    db_path = os.path.join(config.output_folder, f"{search_query}.db")
    Path(config.output_folder).mkdir(parents=True, exist_ok=True)

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS papers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    search_query TEXT,
                    title TEXT UNIQUE,
                    pdf_link TEXT,
                    file_path TEXT,
                    file_hash TEXT,
                    file_content BLOB
                )
            """)
            conn.commit()

            logger.info(f"Searching for papers on '{search_query}'...")
            response_text = _fetch_arxiv_metadata(search_query, max_results)
            papers = _parse_paper_links(response_text)

            logger.info(f"Found {len(papers)} papers. Starting download...")

            downloaded_count = 0
            for title, pdf_link, file_name in papers:
                try:
                    file_path = os.path.join(config.output_folder, file_name)
                    
                    # Check if file exists
                    if os.path.exists(file_path):
                        existing_hash = compute_file_hash(file_path)
                        cursor.execute("SELECT file_hash FROM papers WHERE file_path = ?", (file_path,))
                        row = cursor.fetchone()
                        if row and row[0] == existing_hash:
                            logger.info(f"Skipping already downloaded: {title}")
                            continue

                    # Download & store
                    file_path = _download_paper(title, pdf_link, file_name, config.output_folder)
                    if file_path:
                        file_hash = compute_file_hash(file_path)

                        with open(file_path, "rb") as f:
                            file_content = f.read()

                        cursor.execute("""
                            INSERT OR IGNORE INTO papers (search_query, title, pdf_link, file_path, file_hash, file_content)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, (search_query, title, pdf_link, file_path, file_hash, file_content))
                        conn.commit()

                        downloaded_count += 1
                        time.sleep(2)

                    else:
                        logger.warning(f"Skipping database entry for {title} due to download failure.")

                except sqlite3.Error as e:
                    logger.exception(f"Database error for {title}: {e}")
                    conn.rollback()
                except Exception as e:
                    logger.exception(f"Unexpected error for {title}: {e}")
                    conn.rollback()

        logger.info(f"Download complete! {downloaded_count} papers processed.")
        return papers

    except sqlite3.Error as e:
        logger.exception(f"Database connection error: {e}")
    except Exception as e:
        logger.exception("A general error occurred:")
    return []


def sanitize_filename(title: str) -> str:
    return re.sub(r"[^\w\s-]", "", title).strip().replace(" ", "_")


def compute_file_hash(file_path: str, algorithm: str = "sha256") -> str:
    hash_func = hashlib.new(algorithm)
    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            hash_func.update(chunk)
    return hash_func.hexdigest()


def _fetch_arxiv_metadata(search_query: str, max_results: int) -> str:
    url = f"{config.base_url}search_query=all:{search_query}&start=0&max_results={max_results}"
    logger.info(f"Fetching metadata from: {url}")
    response = requests.get(url)
    response.raise_for_status()
    return response.text


def _parse_paper_links(response_text: str) -> List[Tuple[str, str]]:
    root = ET.fromstring(response_text)
    papers = []
    for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
        title = entry.find("{http://www.w3.org/2005/Atom}title").text
        pdf_link = None

        for link in entry.findall("{http://www.w3.org/2005/Atom}link"):
            if link.attrib.get("title") == "pdf":
                pdf_link = link.attrib["href"] + ".pdf"
                break

        if pdf_link and title:
            file_name = os.path.basename(urlparse(pdf_link).path)
            papers.append((title, pdf_link, file_name))
            logger.info(f"Found paper: {title}, pdf: {pdf_link}")

    return papers


def _download_paper(title, pdf_link, file_name, output_folder) -> str:
    """Downloads a single paper PDF."""
    file_path = os.path.join(output_folder, file_name)

    if os.path.exists(file_path):
        logger.info(f"Skipping already downloaded: {file_name}")
        return file_path

    response = requests.get(pdf_link, stream=True)
    response.raise_for_status()

    with open(file_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=1024):
            file.write(chunk)

    logger.info(f"Downloaded: {title}: {file_name}")
    return file_path

```

### **Code Examples**

Check out the [deepresearch](https://github.com/ernanhughes/deepresearch) for the code used in this post and additional examples.



### References

[Open-source DeepResearch – Freeing our search agents](https://huggingface.co/blog/open-deep-research)

[Introducing deep research](https://openai.com/index/introducing-deep-research/)  

[Try Deep Research and our new experimental model in Gemini, your AI assistant](https://blog.google/products/gemini/google-gemini-deep-research/)

[smolagents tools](https://huggingface.co/docs/smolagents/en/reference/tools)