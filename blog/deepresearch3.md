+++
date = '2025-02-08T18:46:58Z'
draft = false
title = 'DeepResearch Part 3: Getting the best web data for your research'
categories = ['smolagents']
tags = ['smolagents', 'nlp']
+++


### Summary

This post details building a robust web data pipeline using SmolAgents. We'll create tools to retrieve content from various web endpoints, convert it to a consistent format (Markdown), store it efficiently, and then evaluate its relevance and quality using Large Language Models (LLMs). This pipeline is crucial for building a knowledge base for LLM applications.


### Web Data Convertor (`MarkdownConverter`)

We leverage the `MarkdownConverter` class, inspired by the one in [`autogen`](https://github.com/microsoft/autogen/blob/v0.4.4/python/packages/autogen-magentic-one/src/autogen_magentic_one/markdown_browser/mdconvert.py), to handle the diverse formats encountered on the web. This ensures consistency for downstream processing.

`MarkdownConverter` efficiently converts various document formats (HTML, Wikipedia, YouTube content, PDFs, DOCX, XLSX, PPTX, audio, images) into Markdown.  It uses libraries like `mammoth`, `markdownify`, `pdfminer`, `pptx`, `puremagic`, `pydub`, `speech_recognition`, `beautifulsoup4`, `youtube_transcript_api`, and `requests`.


Key features include:

* **Automatic Format Detection:** Uses `puremagic` and file extensions.
* **Format-Specific Conversion:** Handles metadata extraction, content parsing, and Markdown formatting for each format.
* **HTML/Wikipedia Processing:** Extracts relevant content, removes scripts and styles.
* **YouTube Support:** Extracts titles, descriptions, metadata, and transcripts.
* **Office Document Conversion:** Converts DOCX, XLSX, and PPTX, preserving formatting where possible.
* **Audio Conversion:** Converts WAV and MP3, including speech-to-text transcription.
* **Image Conversion:** Extracts metadata and optionally generates descriptions (using a multimodal LLM - consider adding details about this if used).
* **Robust Error Handling:** Includes error handling and logging (add details about logging if implemented).
* **Customizable Markdown:** Uses a custom `_CustomMarkdownify` for flexible Markdown output.
* **Temporary Files:** Manages temporary files for processing.
* **Flexible Input:** Accepts file paths, URLs, and `requests.Response` objects.

This tool simplifies document processing for LLM applications by providing a unified interface and consistent Markdown output.  It's based on the `autogen` repository and is designed for scalability and maintainability.


### Building a web search and storage tool

This tool automates web searching using DuckDuckGo, Google, and Bing, downloads the pages, converts them to Markdown, and stores them in an SQLite database.

**Key Features:**

* **Multi-Search Engine Support:** Queries DuckDuckGo, Google, and Bing.
* **Markdown Conversion:** Converts HTML to Markdown.
* **Structured Database Storage (SQLite):** Stores URLs, filenames, file sizes, file types, content, and timestamps.
* **Duplicate Prevention:** Avoids storing duplicate URLs.
* **Modular Design:** Uses separate classes for configuration, database management, and web searching, making it easier to extend or modify.

**How It Works:**

1. **Configuration:** The `SearchConfig` class manages settings like the chosen search engine, number of results, database path, and API keys.
2. **Database Management:** The `DatabaseManager` class handles database initialization and data insertion, preventing duplicates.
3. **Web Search and Download:** The `WebSearchDownloader` class performs the searches using the appropriate API, downloads the pages, converts them to Markdown, and stores them in the database.
4. **Modular Search Functions:**  The `search_duckduckgo`, `search_google`, and `search_bing` functions abstract the interaction with each search engine's API, allowing for easy addition of more search engines in the future.
5. **Error Handling:** Includes basic error handling for API requests and database operations.

```python

import os
import re
import sqlite3
import requests
from datetime import datetime
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import markdownify
from googleapiclient.discovery import build

class SearchConfig:
    def __init__(self, search_engine="duckduckgo", num_results=10, db_path="downloads.db", 
                 searxng_instance="https://searx.be/api/v1/search", google_api_key=None, google_cse_id=None, bing_api_key=None):
        self.search_engine = search_engine
        self.num_results = num_results
        self.db_path = db_path
        self.searxng_instance = searxng_instance
        self.google_api_key = google_api_key
        self.google_cse_id = google_cse_id
        self.bing_api_key = bing_api_key

class DatabaseManager:
    def __init__(self, db_path):
        self.db_path = db_path
        self._initialize_database()

    def _initialize_database(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS query (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT UNIQUE,
                    datetime TEXT
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS download (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    url TEXT UNIQUE,
                    filename TEXT,
                    file_size INTEGER,
                    file_type TEXT,
                    content TEXT,
                    datetime TEXT
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS query_downloads (
                    query_id INTEGER,
                    download_id INTEGER,
                    FOREIGN KEY (query_id) REFERENCES query(id),
                    FOREIGN KEY (download_id) REFERENCES download(id)
                )
            ''')
            conn.commit()

    def insert_query(self, query):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''INSERT OR IGNORE INTO query (query, datetime) VALUES (?, ?)''',
                           (query, datetime.now().isoformat()))
            conn.commit()
            cursor.execute('''SELECT id FROM query WHERE query = ?''', (query,))
            return cursor.fetchone()[0]

    def insert_record(self, query_id, url, filename, file_size, file_type, content):
        if file_size == 0:
            print(f"Skipping empty file from {url}")
            return
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute('''
                    INSERT INTO download (url, filename, file_size, file_type, content, datetime)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (url, filename, file_size, file_type, content, datetime.now().isoformat()))
                download_id = cursor.lastrowid
                cursor.execute('''INSERT INTO query_downloads (query_id, download_id) VALUES (?, ?)''', (query_id, download_id))
                conn.commit()
            except sqlite3.IntegrityError:
                print(f"URL already exists in the database: {url}")

class WebSearchDownloader:
    def __init__(self, config):
        self.config = config
        self.db_manager = DatabaseManager(config.db_path)

    def search_duckduckgo(self, query):
        """Search DuckDuckGo and return a list of result URLs."""
        with DDGS() as ddgs:
            return [r['href'] for r in ddgs.text(query, max_results=self.config.num_results)]

    def search_google(self, query):
        """Search Google and return a list of result URLs."""
        try:
            service = build("customsearch", "v1", developerKey=self.config.google_api_key)
            result = service.cse().list(q=query, cx=self.config.google_cse_id, num=self.config.num_results).execute()
            return [item['link'] for item in result.get("items", [])]
        except Exception as e:
            print(f"Failed to fetch results from Google: {e}")
            return []

    def search_bing(self, query):
        """Search Bing and return a list of result URLs."""
        try:
            headers = {"Ocp-Apim-Subscription-Key": self.config.bing_api_key}
            params = {"q": query, "count": self.config.num_results}
            response = requests.get("https://api.bing.microsoft.com/v7.0/search", headers=headers, params=params)
            response.raise_for_status()
            results = response.json().get("webPages", {}).get("value", [])
            return [r['url'] for r in results]
        except Exception as e:
            print(f"Failed to fetch results from Bing: {e}")
            return []

    def convert_to_markdown(self, html):
        """Convert HTML content to Markdown format."""
        return markdownify.markdownify(html, heading_style="ATX")

    def sanitize_filename(self, filename):
        """Ensure the filename contains only alphanumeric characters or underscores and is lowercase."""
        filename = re.sub(r'[^a-zA-Z0-9]', '_', filename)
        filename = re.sub(r'_+', '_', filename)
        return filename.lower()

    def download_and_store(self, query_id, url):
        """Download a webpage, convert it to Markdown, store it in the database."""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            content = self.convert_to_markdown(response.text)
            filename = self.sanitize_filename(url.split("/")[-1] or "page") + ".md"
            file_size = len(content.encode("utf-8"))
            file_type = "markdown"
            
            self.db_manager.insert_record(query_id, url, filename, file_size, file_type, content)
            print(f"Stored in database: {filename}")
        except Exception as e:
            print(f"Failed to download and store {url}: {e}")

    def run(self, query):
        """Search the web using the specified engine and store results in the database."""
        query_id = self.db_manager.insert_query(query)
        
        if self.config.search_engine == "google":
            urls = self.search_google(query)
        elif self.config.search_engine == "bing":
            urls = self.search_bing(query)
        else:
            urls = self.search_duckduckgo(query)
        
        for url in urls:
            self.download_and_store(query_id, url)
```

Suggested usage would be to use all the engines. The next section will organize the results and summarize the contents so we can process more data.

The download tool will organize the search results and wont download the same website twice.


---
### Evaluating Web Page Content with LLMs

After collecting and storing the web data, we need to evaluate its quality and relevance. We'll use LLMs for this purpose, combined with embeddings for initial relevance scoring.

Building a Document Evaluation Pipeline
This section describes a pipeline for evaluating the quality and relevance of the downloaded web pages using LLMs.

Key Features:

* **Relevance Scoring with Embeddings**: Uses sentence transformer embeddings to calculate the semantic similarity between the document and the search term.
* **Correctness Evaluation with LLMs**: Prompts an LLM to assess the factual accuracy, logical consistency, and overall quality of the document.
* **Structured Evaluation with JSON**: Uses a JSON schema for consistent and easily parsable evaluation results.
Combined Scoring: Combines relevance and correctness scores for a final ranking.
* **Database Storage**: Stores the evaluation results in a database for easy retrieval and analysis.

#### How It Works:

* **Document Loading and Cleaning**: Loads documents from the database and cleans the Markdown content.
* **Relevance Calculation**: Calculates relevance scores using sentence transformer embeddings.
* **Correctness Evaluation**: Prompts the LLM with a structured prompt to evaluate the document's correctness and generates a JSON response with scores and explanations.
* **Score Combination**: Combines relevance and correctness scores to calculate a final score.
* **Database Storage**: Stores the evaluation results, including the JSON response from the LLM, in the database.

```python
import requests
import re
import openai
import sqlite3
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class Config:
    """Manages configurable parameters."""
    def __init__(self, topic, api_key, model_name='all-MiniLM-L6-v2', llm_models=['gpt-4'], db_path='results.db'):
        self.topic = topic
        self.api_key = api_key
        self.model_name = model_name
        self.llm_models = llm_models if isinstance(llm_models, list) else [llm_models]
        self.db_path = db_path
        openai.api_key = api_key

class MarkdownCleaner:
    @staticmethod
    def clean_markdown(md_text):
        """Remove Markdown syntax and extract clean text."""
        md_text = re.sub(r'\[.*?\]\(.*?\)', '', md_text)  # Remove links
        md_text = re.sub(r'#{1,6}\s*', '', md_text)  # Remove headers
        md_text = re.sub(r'(```.*?```|`.*?`)', '', md_text, flags=re.DOTALL)  # Remove code blocks
        md_text = re.sub(r'\*{1,2}|\_{1,2}', '', md_text)  # Remove bold/italic formatting
        md_text = re.sub(r'>\s*', '', md_text)  # Remove blockquotes
        md_text = re.sub(r'[-+*]\s+', '', md_text)  # Remove bullet points
        md_text = re.sub(r'\d+\.\s+', '', md_text)  # Remove numbered lists
        return md_text.strip()

class EmbeddingModel:
    @staticmethod
    def generate_embeddings(text, model_name="mxbai-embed-large", base_url="http://localhost:11434"):
        try:
            print(f"Model: {model_name}")
            url = f"{base_url}/api/embed"
            response = requests.post(url, json={"input": text, "model": model_name})
            response.raise_for_status()
            return response.json()["embeddings"]
        except requests.RequestException as e:
            logging.error(f"Embedding generation failed: {e}")
            return None

class ChatModel:
    @staticmethod
    def chat(prompt, model_name="llama3.2", base_url="http://localhost:11434"):
        try:
            url = f"{base_url}/api/chat"
            data = {
                "model": model_name,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "stream": False
            }
            response = requests.post(url, json=data)
            if response.status_code == 200:
                return response.json()["message"]["content"]
            else:
                logging.error(f"Failed to generate response. Status code: {response.status_code}")
                return None
        except requests.ConnectionError:
            logging.error("Failed to connect to the Ollama server.")
            return None
        except json.JSONDecodeError:
            logging.error("Failed to parse JSON response.")
            return None
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            return None

class DocumentEvaluator:
    def __init__(self, config):
        self.config = config
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize the database table to store results."""
        conn = sqlite3.connect(self.config.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS document_scores (
                filename TEXT,
                model_name TEXT,
                relevance_score REAL,
                correctness_score REAL,
                final_score REAL,
                evaluation_json TEXT,
                PRIMARY KEY (filename, model_name)
            )
        ''')
        conn.commit()
        conn.close()
    
    def load_documents_from_db(self):
        """Load documents from the downloads database table."""
        conn = sqlite3.connect(self.config.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT filename, url FROM download")
        documents = cursor.fetchall()
        conn.close()
        return [(doc[0], MarkdownCleaner.clean_markdown(doc[1])) for doc in documents]
    
    def compute_relevance(self, documents):
        """Compute relevance scores using embeddings."""
        doc_texts = [doc[1] for doc in documents]
        doc_embeddings = []
        for doc in doc_texts:
            doc_embeddings.append(EmbeddingModel.generate_embeddings(doc))
        topic_embedding = EmbeddingModel.generate_embeddings('Cellular Automata texture generation')[0]
        scores = cosine_similarity([topic_embedding], doc_embeddings)[0]
        relevance_scores = {doc[0]: score for doc, score in zip(documents, scores)}
        return relevance_scores
    
    def evaluate_correctness_with_llm(self, document_text, model):
        """Use an LLM to evaluate correctness based on factual accuracy and logical consistency, returning structured JSON."""
        prompt = f"""
        You are an expert reviewer evaluating a document's correctness and relevance to the topic "{self.config.topic}".  
        Provide a structured JSON response containing scores and explanations for each criterion below.

        **Evaluation Criteria (1-10 scale, 1=very poor, 10=excellent):**
        * keyword_presence
        * content_focus
        * information_quality
        * user_intent
        * factual_accuracy
        * logical_consistency
        * clarity_and_grammar

        Return JSON with the format:
        ```json
        {{
            "scores": {{
                "keyword_presence": <score>,
                "content_focus": <score>,
                "information_quality": <score>,
                "user_intent": <score>,
                "factual_accuracy": <score>,
                "logical_consistency": <score>,
                "clarity_and_grammar": <score>
            }},
            "overall_relevance": <score>,
            "summary": "<Overall evaluation summary>"
        }}
        ```
        
        Document:
        ```
        {document_text}
        ```
        """
        result_json = ChatModel.chat(model, prompt)
        try:
            evaluation = json.loads(result_json)
            final_score = evaluation.get("overall_relevance", 5)
        except json.JSONDecodeError:
            evaluation = {}
            final_score = 5
        
        return final_score, json.dumps(evaluation)
    
    def rank_documents(self):
        """Rank documents based on relevance and correctness."""
        documents = self.load_documents_from_db()
        relevance_scores = self.compute_relevance(documents)
        
        conn = sqlite3.connect(self.config.db_path)
        cursor = conn.cursor()
        
        for model in self.config.llm_models:
            correctness_scores = {}
            evaluations = {}
            for filename, text in documents:
                correctness_scores[filename], evaluations[filename] = self.evaluate_correctness_with_llm(text, model)
            
            for filename in relevance_scores.keys():
                relevance = relevance_scores[filename]
                correctness = correctness_scores[filename]
                final_score = (relevance * 5) + (correctness * 5)
                cursor.execute('''
                    INSERT INTO document_scores (filename, model_name, relevance_score, correctness_score, final_score, evaluation_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(filename, model_name) DO UPDATE SET
                        relevance_score=excluded.relevance_score,
                        correctness_score=excluded.correctness_score,
                        final_score=excluded.final_score,
                        evaluation_json=excluded.evaluation_json
                ''', (filename, model, relevance, correctness, final_score, evaluations[filename]))
        
        conn.commit()
        conn.close()
```



---

