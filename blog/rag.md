+++
date = '2025-01-14T14:40:59Z'
title = 'Rag: Retrieval-Augmented Generation '
categories = ['AI', 'RAG', 'smolagents']
tag = ['rag'] 
+++

## **Summary**

Retrieval-Augmented Generation (RAG) is a powerful technique that enhances large language models (LLMs) by allowing them to use external knowledge sources.

An **Artificial Intelligence (AI)** system consists of components working together to apply knowledge learned from data. Some common components of those systems are:

- **Large Language Model (LLM)**: Typically the core component of the system, often there is more than one. These are large models that have been trained on massive amounts of data and can make intelligent predictions based on their training.

- **RAG**: Used to enhance LLMs with custom or external knowledge. This knowledge can refine the LLM's behavior, improving its performance in specific tasks.

- **[Agents]({{< ref "/post/agents.md" >}} "Agents")**: Processes that tie the whole system together. Often, there will be a manager-type agent who serves as the interface for the entire system.

In this post, I will show:
1. A [naive](#building-a-naive-rag-system-in-python) (simple) RAG solution.
2. An [agentic](#agent-rag-using-smolagents) RAG solution.

---

## **What is Retrieval-Augmented Generation (RAG)?**

RAG is a hybrid approach that combines **retrieval-based** methods with **generation-based** methods.

- **Generative models** like LLMs generate text based solely on their pre-existing knowledge learned during training. This means that their knowledge can become outdated over time.
  
- **Retrieval-based models** search through a knowledge base to find relevant information and then generate text based on that retrieved data. This allows the system to incorporate up-to-date information.

The RAG approach integrates both strategies: when a query is given, the model first **retrieves relevant documents** (from a prebuilt database or search engine), and then uses this retrieved information to generate a more accurate and contextually appropriate response.

### **Key Benefits of RAG**:
- **New Information**: As things change, RAG allows your system to adapt by fetching the most current information from external sources.
  
- **Improved Accuracy**: By incorporating up-to-date information from external sources, RAG leads to better responses, enhancing the relevance and accuracy of generated answers.

- **Reduced Hallucination**: Traditional models sometimes generate text that sounds plausible but is factually incorrect. RAG mitigates this issue by relying on retrieved documents. Furthermore, you can design the system so that it only generates responses using the retrieved information, ensuring correctness.

- **Flexibility**: You can use custom or external databases to enhance the model’s knowledge base, making it adaptable to specific applications, industries, or niches.

---

## **How RAG Works**

At a high level, the RAG system follows these steps:

1. **Querying**: The input query is given to the retrieval module (typically a search engine or database).
2. **Document Retrieval**: The retrieval system fetches the top-k relevant documents.
3. **Document Embedding**: The retrieved documents are transformed into embeddings (numerical representations of text) using a model. You can use the [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard) to choose a good model for generating embeddings. For this post, I chose to use Ollama, and after reviewing the [Ollama embedding models](https://ollama.com/search?c=embedding), I selected the model **bge-m3**, which is ranked highly on the MTEB leaderboard.

4. **Text Generation**: These embeddings are then used as context for an LLM to generate the final output.

The retrieval system is generally based on techniques like **semantic search**, where the model searches for documents that are semantically similar to the query, improving the accuracy of the retrieved information.

---

## **Building a Naive RAG System in Python**

In this section, we’ll create a basic RAG system using Python. We’ll generate embeddings using a model and use [FAISS](https://github.com/facebookresearch/faiss) for efficient similarity search.

```python
def generate_embeddings(document_list: list, model_name: str = "bge-m3:latest") -> list[torch.FloatTensor]:
    embeddings = []
    for document in document_list:
        embedding_list = _generate_embeddings(document, model_name)
        embedding = torch.FloatTensor(embedding_list)
        embeddings.append(embedding)
    assert len(embeddings) == len(document_list)
    return torch.stack(embeddings, dim=0)

def _generate_embeddings(document: str, model_name: str = "bge-m3:latest") -> list[float]:
    """Generate embeddings for the given text using the specified model."""
    try:
        logging.debug("Generating embeddings for {}".format(document))
        # Send a POST request to generate embeddings
        url = f"{OLLAMA_BASE_URL}/api/embeddings"
        data = {
            "prompt": document,
            "model": model_name
        }
        response = requests.post(url, json=data)
        
        if response.status_code == 200:
            embeddings = response.json()["embedding"]
            return embeddings
        else:
            logging.error(f"Failed to generate embeddings. Status code: {response.status_code}")
            return []
    
    except requests.ConnectionError:
        logging.error("Failed to connect to the Ollama server. Make sure it is running locally and the URL is correct.")
        return []
    except json.JSONDecodeError:
        logging.error("Failed to parse JSON response from Ollama server.")
        return []
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return []
```

This function uses the **Ollama** API to generate embeddings for a list of documents.

### Example documents to search from:
```python
documents = [
    "Python is a programming language that lets you work quickly.",
    "Machine learning automates model building through data analysis.",
    "Artificial Intelligence (AI) is intelligence demonstrated by machines.",
    "Natural language processing (NLP) is a field of AI that focuses on human language interaction.",
]
```

After generating the embeddings, we will build a **FAISS index** to perform a fast similarity search.

```python
# Generate embeddings for documents
document_embeddings = generate_embeddings(documents)
logging.info(f"Generated Document Embeddings: {document_embeddings.shape}")

# Build a FAISS index for fast similarity search
index = faiss.IndexFlatL2(document_embeddings.shape[1])

def retrieve_documents(query, top_k=2):
    query_embedding = generate_embeddings([query])
    _, indices = index.search(np.array(query_embedding), top_k)
    return [documents[i] for i in indices[0]]

query = "Tell me about artificial intelligence"
retrieved_docs = retrieve_documents(query)
print("Retrieved Documents: {}".format(retrieved_docs))
```

---

### **Step 3: Text Generation Using Ollama**

Now, let’s generate the answer based on the retrieved documents:

```python
def chat_with_ollama(prompt, model_name="llama3.2:latest"):
    try:
        url = f"{OLLAMA_BASE_URL}/api/chat"
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
```

---

### **Step 4: Putting It All Together**

Now that we have the retrieval and generation functions, let’s integrate them into a complete RAG pipeline.

```python
def format_docs(docs):
    formatted_docs = ""
    for i, doc in enumerate(docs):
        formatted_docs += f"{i+1}. {doc}\n"
    return formatted_docs

def generate_answer(query, retrieved_docs):
    context = f"Using this information:\n { format_docs(retrieved_docs) }\n Can you answer this question: {query}."
    return chat_with_ollama(context)

def rag_system(query):
    retrieved_docs = retrieve_documents(query)
    generated_answer = generate_answer(query, retrieved_docs)
    return generated_answer

# Test the RAG system
query = "What is Natural Language Processing?"
answer = rag_system(query)
print(f"Final Answer:\n{answer}")
```

---

## **Agent RAG Using smolagents**

This example shows how agents can be used with RAG to further enhance its capabilities. We adapt an example from the [smolagents repository](https://github.com/huggingface/smolagents/blob/main/examples/rag.py).

```python
import datasets
knowledge_base = datasets.load_dataset("m-ric/huggingface_doc", split="train")
filtered_knowledge_base = knowledge_base.filter(lambda row: row["source"].startswith("huggingface/transformers"))

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever

# Process documents for the retriever
source_docs = [
    Document(page_content=doc["text"], metadata={"source": doc["source"].split("/")[1]})
    for doc in filtered_knowledge_base
]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs_processed = text_splitter.split_documents(source_docs)

from smolagents import Tool

class RetrieverTool(Tool):
    name = "retriever"
    description = "Uses semantic search to retrieve relevant documents."
    inputs = {"query": {"type": "string", "description": "Search query."}}
    output_type = "string"
    
    def __init__(self, docs, **kwargs):
        super().__init__(**kwargs)
        self.retriever = BM25Retriever.from_documents(docs, k=10)

    def forward(self, query: str) -> str:
        docs = self.retriever.invoke(query)
        return "\n".join([f"Document {i}: {doc.page_content}" for i, doc in enumerate(docs)])

from smolagents import CodeAgent, LiteLLMModel

model = LiteLLMModel(model_id="ollama/qwen2.5-coder:14b", api_base="http://localhost:11434")
retriever_tool = RetrieverTool(docs_processed)
agent = CodeAgent(tools=[retriever_tool], model=model, max_steps=4, verbosity_level=2)

agent_output = agent.run("For a transformers model training, which pass is slower?")
print("Final output:", agent_output)
```

---

## **Conclusion**

In this post, we’ve built both a naive and agent-based RAG system in Python. By combining the strengths of retrieval and generation-based models, RAG can significantly improve the accuracy of AI systems in answering complex queries by providing real-time, up-to-date information.

### **Key Takeaways**:
- **RAG enhances the generative capabilities of LLMs** by retrieving relevant documents to augment the output.
- **FAISS and Ollama** are excellent tools for implementing semantic search, which is central to RAG.
- You can use **Ollama** or any other generative model to synthesize the retrieved information into a coherent response.
- **Agents** can be used to further enhance the RAG process by organizing tasks and interactions more intelligently.

## **Code Examples**

Check out the [RAG notebooks](https://github.com/ernanhughes/rag-notebooks) for the code used in this post and additional examples.

