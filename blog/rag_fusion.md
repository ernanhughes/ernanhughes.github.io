+++
date = '2025-02-23T07:25:24Z'
draft = true
title = 'Rag_fusion'
+++

Here's a Python implementation that stores both the generated hierarchical queries and the corresponding prompts in an SQLite database.

### **Step 1: Set Up SQLite Database**
```python
import sqlite3
import re

# Create or connect to an SQLite database
conn = sqlite3.connect("queries.db")
cursor = conn.cursor()

# Create table to store prompts and generated queries
cursor.execute("""
CREATE TABLE IF NOT EXISTS queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    original_query TEXT,
    prompt TEXT,
    generated_query TEXT
)
""")
conn.commit()
```

### **Step 2: Modify Your Function to Store Data**
```python
class QueryGenerator:
    def __init__(self, llm):
        self.llm = llm  # Assume `llm` is a function that generates text from a prompt
        self.regex = r"\d+\.\s(.+)"  # Regex to extract numbered results

    def store_query(self, original_query, prompt, generated_text):
        """Store the generated query in the SQLite database."""
        matches = re.findall(self.regex, generated_text)
        for query in matches:
            cursor.execute(
                "INSERT INTO queries (original_query, prompt, generated_query) VALUES (?, ?, ?)",
                (original_query, prompt, query),
            )
        conn.commit()

    def generate_hierarchical_queries(self, original_query):
        # Step 1: Generate broad categories
        category_prompt = f"""Generate broad categories for: {original_query} 
                              Please return the categories in numbered format:
                              1. First Category
                              2. Second Category
                              3. Third Category
                           """
        generated_text = self.llm(category_prompt)
        self.store_query(original_query, category_prompt, generated_text)

        categories = re.findall(self.regex, generated_text)

        all_queries = []
        for category in categories:
            # Step 2: Generate specific queries for each category
            query_prompt = f"""Generate specific search queries for the category: {category}
                               Please return the queries in numbered format:
                               1. First query
                               2. Second query
                               3. Third query
                            """
            generated_text = self.llm(query_prompt)
            self.store_query(original_query, query_prompt, generated_text)

            queries = re.findall(self.regex, generated_text)
            all_queries.extend(queries)

        return all_queries
```

### **Step 3: Querying the Database**
To retrieve stored queries:
```python
def get_stored_queries():
    cursor.execute("SELECT * FROM queries")
    rows = cursor.fetchall()
    for row in rows:
        print(row)

get_stored_queries()
```

### **How It Works:**
1. The `generate_hierarchical_queries` function:
   - Generates broad categories.
   - Stores the prompt and generated categories in the database.
   - Iterates over each category to generate specific search queries.
   - Stores those as well.
2. The `store_query` function inserts data into the SQLite database.
3. You can retrieve all stored queries using `get_stored_queries()`.


---

To implement **Cross-Encoder Re-ranking**, you have several options depending on the specific requirements of your task, computational resources, and preferred frameworks. Here’s a breakdown of the approaches:

---

### **1. Using Pre-Trained Cross-Encoders from Sentence Transformers**
The `sentence-transformers` library provides pre-trained cross-encoders for ranking documents, which is an efficient way to implement re-ranking.

#### **Installation**
```bash
pip install sentence-transformers
```

#### **Example: Using a Cross-Encoder for Re-Ranking**
```python
from sentence_transformers import CrossEncoder

# Load a pre-trained cross-encoder model
model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Define query and candidate responses (or documents)
query = "What are the best practices for deploying transformers?"
documents = [
    "Deploying transformers efficiently requires proper memory management.",
    "Best practices include quantization and model pruning.",
    "Transformers have applications in NLP, vision, and more."
]

# Create input pairs for cross-encoder ranking
input_pairs = [(query, doc) for doc in documents]

# Get similarity scores
scores = model.predict(input_pairs)

# Rank documents by score (higher is better)
ranked_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

# Display ranked results
for rank, (doc, score) in enumerate(ranked_docs, 1):
    print(f"{rank}. {doc} (Score: {score:.4f})")
```

---
### **2. Using a Hugging Face Model for Cross-Encoding**
Hugging Face provides many cross-encoder models that can be used for ranking.

#### **Example: Using a Transformer Model Directly**
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load model and tokenizer
model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def get_ranking_score(query, document):
    inputs = tokenizer(query, document, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        scores = model(**inputs).logits
    return scores.item()

# Rank documents
query = "How to optimize transformers?"
documents = ["Use mixed precision training.", "Fine-tune the model with low learning rates.", "Data augmentation improves performance."]
ranked = sorted(documents, key=lambda doc: get_ranking_score(query, doc), reverse=True)

# Print ranked results
for rank, doc in enumerate(ranked, 1):
    print(f"{rank}. {doc}")
```

---

### **3. Implementing a Custom Cross-Encoder**
If you want to train a custom cross-encoder, you can fine-tune a transformer model.

#### **Fine-Tuning a Cross-Encoder with Sentence Transformers**
```python
from sentence_transformers import SentenceTransformer, LoggingHandler, InputExample, losses, models, datasets
from torch.utils.data import DataLoader

# Define transformer model
model = CrossEncoder("distilroberta-base", num_labels=1)

# Create training data
train_samples = [
    InputExample(texts=["What is AI?", "AI is a field of computer science."], label=1.0),
    InputExample(texts=["What is NLP?", "NLP is a subset of AI focusing on language."], label=0.9),
    InputExample(texts=["How to cook pasta?", "Buy a laptop to start coding."], label=0.1)
]

train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=8)
train_loss = losses.RankingLoss(model)

# Train the model
model.fit(train_dataloader=train_dataloader, epochs=1, warmup_steps=100)
```

---
### **4. Using Cross-Encoders in a Pipeline with BM25**
If you have a large document collection, you can first retrieve relevant documents using **BM25** (a classical ranking function) and then apply a cross-encoder for re-ranking.

#### **BM25 + Cross-Encoder Hybrid**
```python
from rank_bm25 import BM25Okapi

# Example corpus
documents = ["AI is the future.", "Machine learning helps AI.", "Neural networks power AI models."]
tokenized_corpus = [doc.split() for doc in documents]

# Initialize BM25
bm25 = BM25Okapi(tokenized_corpus)

# Retrieve initial candidates
query = "AI and machine learning"
tokenized_query = query.split()
bm25_scores = bm25.get_scores(tokenized_query)

# Sort by BM25 score
bm25_ranked = sorted(zip(documents, bm25_scores), key=lambda x: x[1], reverse=True)

# Apply Cross-Encoder Re-ranking
cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
reranked = sorted(bm25_ranked, key=lambda x: cross_encoder_model.predict([(query, x[0])]), reverse=True)

# Display final ranked results
for rank, (doc, score) in enumerate(reranked, 1):
    print(f"{rank}. {doc} (Score: {score:.4f})")
```

---
### **Which Approach to Choose?**
| Approach | Pros | Cons |
|----------|------|------|
| **Pre-trained Cross-Encoders (Sentence Transformers)** | Easy to use, optimized models available | Limited customization |
| **Hugging Face Transformers** | More control over inference | Requires handling tokenization and scoring manually |
| **Fine-Tuning a Cross-Encoder** | Customizable for specific tasks | Needs labeled training data and compute resources |
| **BM25 + Cross-Encoder** | Efficient for large-scale retrieval | Two-step process may add latency |

