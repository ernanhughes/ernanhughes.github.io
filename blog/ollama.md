+++
date = '2025-01-06T19:52:16Z'
draft = true
title = 'Ollama: The local LLM solution'
+++

### **Using Ollama**

---

### **Introduction**

Ollama is the best platform for running, managing, and interacting with Large Language Models (LLM) models locally. 
For Python programmers, Ollama offers seamless integration and robust features for querying, manipulating, and deploying LLMs. 
In this post I will explore how Python developers can leverage Ollama for powerful and efficient AI-based workflows.

---

### **1. What is Ollama?**

Ollama is a tool designed to enable local hosting and interaction with LLMs. Unlike cloud-based APIs, Ollama prioritizes privacy and speed by running models directly on your machine. Key benefits include:

- **Local Model Execution**: No internet connection is needed for most interactions.
- **Privacy**: Data never leaves your machine.
- **Python Integration**: Easily interact with Ollama's models through Python scripts.
- **Access to many powerful Models**: With Ollama you have access to all open models.
- **No Cost Solutions**: Allows you to build AI solutions with no cost. There are no fees for using Ollama Models.

For Python programmers, Ollama unlocks possibilities for integrating AI models into projects without relying on external services.

---

### **2. Installing and Setting Up Ollama**

#### **Step 1: Install Ollama**
Download Ollama from its official website ([ollama.com](https://ollama.com)) and follow the installation instructions for your operating system.

#### **Step 2: Verify Installation**
After installation, verify that Ollama is working by running:
```bash
ollama --version
```

#### **Step 3: Install Python SDK**
Ollama can be accessed from Python using REST APIs or third-party libraries. Ensure you have Python installed and create a virtual environment for your project:
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

Install `requests` for interacting with the Ollama API:
```bash
pip install requests
```

---

### **3. Chat with an Ollama Model**

Here’s how to interact with an Ollama model using Python:

```python
import requests

# Ollama server URL
def chat_with_ollama(text, model_name):
    """Chat with Ollama."""
    try:
        url = f"{OLLAMA_BASE_URL}/api/chat"
        data = {
            "prompt": text,
            "model": model_name,
            "stream": False
        }
        response = requests.post(url, json=data)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Parse the JSON response
            embeddings = response.json()
            print("Chat Response:")
            pretty_json = json.dumps(embeddings, indent=4)
            print(pretty_json)
            return embeddings
        else:
            print(f"Failed to generate embeddings. Status code: {response.status_code}")
            print("Response:", response.text)
            return None
    
    except requests.ConnectionError:
        print("Failed to connect to the Ollama server. Make sure it is running locally and the URL is correct.")
        return None
    except json.JSONDecodeError:
        print("Failed to parse JSON response from Ollama server.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
# Example usage
text = "Why is recursion in python?"
model_name = "llama3.2"
res = chat_with_ollama(text, model_name)
print(res)
```


#### **Output Example**:
```
Recursion in Python is a process where a function calls itself to solve smaller instances of a problem...
```

---

### **4. List Ollama models installed**

```Python
import requests
import json

OLLAMA_BASE_URL = "http://localhost:11434"

def filter_response(json_data: str, filter_keys: list[str]):
    for model in json_data:
        print(f"| {model['name']} | {model['model']} |")
    
def list_ollama_models():
    """List installed models from the Ollama server."""
    try:
        url = f"{OLLAMA_BASE_URL}/api/tags"
        response = requests.get(url)
        if response.status_code == 200:
            print("Installed Ollama Models:")
            filtered_models = filter_response(response.json()["models"], ['name', 'tags'])
            print(filtered_models)
            return filtered_models
        else:
            print(f"Failed to retrieve models. Status code: {response.status_code}")
            print("Response:", response.text)
            return []
    except requests.ConnectionError:
        print("Failed to connect to the Ollama server. Make sure it is running locally and the URL is correct.")
        return []
    except json.JSONDecodeError:
        print("Failed to parse JSON response from Ollama server.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

list_ollama_models()
```
The result will be something like this

```
Installed Ollama Models:
| phi:latest | phi:latest |
| nomic-embed-text:latest | nomic-embed-text:latest |
| phi3:latest | phi3:latest |
| llama3.2:latest | llama3.2:latest |
| mxbai-embed-large:latest | mxbai-embed-large:latest |
| mistral:latest | mistral:latest |

```

### **5. Show information on a local model**

This will give you detailed information on the model. 
Where I have found this useful is when calling the model using  prompts with parameters.

```Python

def show_model(model_name):
    """Show model details."""
    try:
        url = f"{OLLAMA_BASE_URL}/api/show"
        data = {
            "model": model_name
        }
        response = requests.post(url, json=data)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Parse the JSON response
            model_details = response.json()
            print("Show Model Response:")
            pretty_json = json.dumps(model_details, indent=4)
            print(pretty_json)
            return model_details
        else:
            print(f"Failed to generate embeddings. Status code: {response.status_code}")
            print("Response:", response.text)
            return None
    
    except requests.ConnectionError:
        print("Failed to connect to the Ollama server. Make sure it is running locally and the URL is correct.")
        return None
    except json.JSONDecodeError:
        print("Failed to parse JSON response from Ollama server.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
# Example usage
model_name = "llama3.2"
res = show_model(model_name)
print(res)
```

Will generate the following response:

```json

{
    "license": "LLAMA 3.2 COMMUNITY LICENSE AGREEMENT Llama 3.2",
    "modelfile": "# Modelfile generated by \"ollama show\"","parameters": "stop                           \"<|start_header_id|>\"\nstop                           \"<|end_header_id|>\"\nstop                           \"<|eot_id|>\"",
    "details": {
        "parent_model": "",
        "format": "gguf",
        "family": "llama",
        "families": [
            "llama"
        ],        ],
        "parameter_size": "3.2B",
        "quantization_level": "Q4_K_M"
    },
    "model_info": {
        "general.architecture": "llama",
        "general.basename": "Llama-3.2",
        "general.file_type": 15,
        "general.finetune": "Instruct",
        "general.languages": [
            "en",
            "de",
            "fr",
    
    }
}
```

---

### **6. Generate Embeddings**

Language model embeddings are a way of representing text as numerical vectors (arrays of numbers) in a mathematical space. These vectors capture the meaning and relationships between the text elements in a way that computers can process and analyze.

It like having multiple database indexes on the text where the indexes can be on anything we want. This allows us to group similar items together. We then use these groupings to extract more information from the data.

```Python
def generate_embeddings(text, model_name):
    """Generate embeddings for the given text using the specified model."""
    try:
        # Send a POST request to generate embeddings
        url = f"{OLLAMA_BASE_URL}/api/embeddings"
        data = {
            "prompt": text,
            "model": model_name
        }
        response = requests.post(url, json=data)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Parse the JSON response
            embeddings = response.json()
            print("Embeddings:")
            pretty_json = json.dumps(embeddings, indent=4)
            print(pretty_json)
            return embeddings
        else:
            print(f"Failed to generate embeddings. Status code: {response.status_code}")
            print("Response:", response.text)
            return None
    
    except requests.ConnectionError:
        print("Failed to connect to the Ollama server. Make sure it is running locally and the URL is correct.")
        return None
    except json.JSONDecodeError:
        print("Failed to parse JSON response from Ollama server.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
# Example usage
text = "Hello, world!"
model_name = "nomic-embed-text"
res = generate_embeddings(text, model_name)
print(res)
```

The response will be something like:

```
Embeddings:
{
    "embedding": [
        0.3587897717952728,
        -0.019634757190942764,
        -3.578793525695801,
    ....
        -1.039941668510437,
        0.17351241409778595
    ]
}      
```

### **7. Pull a new Model**

```Python

def pull_model(model_name):
    """Pull ollama model."""
    try:
        url = f"{OLLAMA_BASE_URL}/api/pull"
        data = {
            "name": model_name
        }
        response = requests.post(url, json=data)
        
        if response.status_code == 200:
            print(f"Pull Model Successful! Response: {response}")
        else:
            print(f"Failed to pull model. Status code: {response.status_code}")
            print("Response:", response.text)
            return None
    
    except requests.ConnectionError:
        print("Failed to connect to the Ollama server. Make sure it is running locally and the URL is correct.")
        return None
    except json.JSONDecodeError:
        print("Failed to parse JSON response from Ollama server.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
# Example usage
model_name = "mistral-small"
res = pull_model(model_name)
```

The result if successful will be like:

```
Pull Model Successful! Response: <Response [200]>
```

### **8. Delete a model**

```Python
def delete_model(model_name):
    """Delete ollama model."""
    try:
        url = f"{OLLAMA_BASE_URL}/api/delete"
        data = {
            "name": model_name
        }
        response = requests.post(url, json=data)
        
        if response.status_code == 200:
            print(f"Delete Model Successful! Response: {response}")
        else:
            print(f"Failed to delete model. Status code: {response.status_code}")
            print("Response:", response.text)
            return None
    
    except requests.ConnectionError:
        print("Failed to connect to the Ollama server. Make sure it is running locally and the URL is correct.")
        return None
    except json.JSONDecodeError:
        print("Failed to parse JSON response from Ollama server.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
# Example usage
model_name = "mistral-small"
res = delete_model(model_name)

```

The result if successful will be:

```

```

### **9. Advanced Topics**

#### **A. Streaming Responses**
To handle large outputs, Ollama supports streaming responses. Here’s an example:
```python
def query_ollama_stream(model, prompt):
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True
    }
    
    with requests.post(OLLAMA_URL, json=payload, headers=headers, stream=True) as response:
        if response.status_code == 200:
            for chunk in response.iter_content(chunk_size=None):
                print(chunk.decode("utf-8"), end="")
        else:
            raise Exception(f"Error: {response.status_code}, {response.text}")

# Example usage
query_ollama_stream("llama2", "Write a Python script for a to-do list application.")
```

#### **B. Fine-Tuning and Custom Models**
While Ollama doesn’t directly support model fine-tuning, you can define custom prompts and system behaviors to tailor the model’s output. Use the `system` message role:
```python
payload = {
    "model": "llama2",
    "messages": [
        {"role": "system", "content": "You are an expert Python programmer."},
        {"role": "user", "content": "How do I implement a decorator in Python?"}
    ]
}
```

#### **C. Batch Processing**
For handling multiple queries at once, Python's `concurrent.futures` can be used:
```python
from concurrent.futures import ThreadPoolExecutor

def batch_query_ollama(model, prompts):
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda p: query_ollama(model, p), prompts))
    return results

# Example usage
prompts = ["Explain recursion.", "What is a Python generator?", "How do I use asyncio?"]
responses = batch_query_ollama("llama2", prompts)
for resp in responses:
    print(resp)
```

---

### **10. Deploying Ollama Models in Applications**

#### **A. Flask Integration**
Create a Flask API endpoint that serves responses from an Ollama model:
```python
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

@app.route("/api/query", methods=["POST"])
def query():
    data = request.json
    model = data.get("model")
    prompt = data.get("prompt")
    try:
        response = query_ollama(model, prompt)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5000)
```

#### **B. Deploying with Docker**
Wrap your application in a Docker container alongside Ollama for portable deployments.

---

### **11. Best Practices for Python Programmers**

1. **Local Model Management**:
   - Periodically update your models for the latest improvements.
   - Use lightweight models for quick prototyping and heavier models for production.

2. **Error Handling**:
   - Always handle connection errors and model-specific issues gracefully.

3. **Streaming and Memory Efficiency**:
   - Stream results for large outputs to avoid blocking your application.

4. **System Resource Monitoring**:
   - Monitor system resources when running large models locally to prevent overloading your machine.

---


## Code examples

[Ollama notebooks](https://github.com/ernanhughes/ollama-notebooks)

has some example code for using Ollama.
