+++
date = '2025-02-24T14:13:26Z'
draft = false
title = 'LiteLLM: A Lightweight Wrapper for Multi-Provider LLMs'
categories = ['LiteLLM', 'SmartAnswer']
tags = ['litellm', 'SmartAnswer']
+++

## Summary

In this post I will cover [**LiteLLM**](https://github.com/BerriAI/litellm). I used it for my implementation of [Textgrad]({{< relref "post/textgrad.md" >}}) also it was using in blog posts I did about [Agents]({{< relref "post/agents.md" >}}).

Working with multiple LLM providers is painful. Every provider has its own API, requiring custom integration, different pricing models, and maintenance overhead.
LiteLLM solves this by offering a single, unified API that allows developers to switch between OpenAI, Hugging Face, Cohere, Anthropic, and others without modifying their code.

If a provider becomes too expensive or does not support the functionality you need you can switch them out for something new.

This approach allows you to focus on your custom code and let it take care of the specifics of interfacing to different LLM providers. 

---

## Why Use LiteLLM?

### 1. **Unified API for Multiple Providers**
LiteLLM provides a consistent interface to interact with multiple LLM APIs, eliminating the need to write separate code for each provider.

### 2. **Cost Optimization**
It allows automatic fallback to cheaper or faster models when necessary, optimizing API costs and performance.

### 3. **Seamless Model Switching**
With LiteLLM, switching from one model provider to another is as simple as changing a parameter.

### 4. **Load Balancing and Routing**
LiteLLM supports model load balancing, routing requests across multiple providers for improved efficiency and availability.

### 5. **Custom Endpoints**
You can define and use custom API endpoints, making LiteLLM a great tool for self-hosted models.

---

## Getting Started with LiteLLM

### Basic Usage
To use LiteLLM, initialize it with your preferred model provider:

```python
from litellm import completion

messages = [{ "content": "There's a llama in my garden 😱 What should I do?","role": "user"}]

response = completion(
    model="ollama/llama3.2",
    messages=[{ "content": "Hello, how are you?","role": "user"}],
    stream=False
)

print(response)
```

```
I'm just a language model, so I don't have feelings or emotions like humans do, but thank you for asking! How can I assist you today?
```

### Using Multiple Providers
You can easily switch between different model providers:

```python
from litellm import completion


response = completion(
    model="gpt-4o",  # Using OpenAI
    messages=[{"role": "user", "content": "Summarize the latest AI research trends."}]
)
print(response["choices"][0]["message"]["content"])
```

```
As of the latest research trends, several key areas have garnered significant attention in the field of artificial intelligence:

1. **Generative AI and Large Language Models (LLMs):** The development and application of generative AI, particularly LLMs like OpenAI's GPT series and 
...
```


---
## Generate embeddings

```python
from litellm import embedding

# I called this inside a notebook
import nest_asyncio
nest_asyncio.apply()


response = embedding(
    model='ollama/nomic-embed-text',
    api_base="http://localhost:11434",
    input=["good morning from litellm"],
    stream=False
)

print(response)
```

```
EmbeddingResponse(model='ollama/nomic-embed-text', data=[{'object': 'embedding', 'index': 0, 'embedding': [...]}], 
object='list', usage=Usage(completion_tokens=6, prompt_tokens=6, total_tokens=6, 
completion_tokens_details=None, prompt_tokens_details=None))
```

---


## Advanced Features

### 1. **Fallback Mechanism**
If one provider fails, LiteLLM can automatically fallback to another:

```python
response = litellm.completion(
    model=["gpt-4", "claude-v1", "mistral-7b"]  # Tries models in sequence
)
```

### 2. **Streaming Responses**
LiteLLM supports streaming responses, reducing response latency:

```python
for chunk in litellm.completion(
    model="gpt-4",
    messages=[{"role": "user", "content": "Tell me a joke."}],
    stream=True
):
    print(chunk["choices"][0]["message"]["content"], end="")
```

### 3. **Batch Requests**
Send multiple queries simultaneously for efficiency:

```python
responses = litellm.batch(
    model="gpt-4",
    messages=[
        [{"role": "user", "content": "Define AI."}],
        [{"role": "user", "content": "What is the speed of light?"}]
    ]
)

for r in responses:
    print(r["choices"][0]["message"]["content"])
```

### 4. **Custom Endpoints for Self-Hosted Models**
If you're running an open-source LLM on your own infrastructure, you can integrate it with LiteLLM:

LiteLLM supports self-hosted models via custom endpoints. You can register local instances of models like Llama.cpp or GPTQ by providing an HTTP endpoint.

```python
litellm.register_custom_endpoint(
    name="my_local_model",
    endpoint="http://localhost:5000/v1/completions"
)

response = litellm.completion(model="my_local_model", messages=[{"role": "user", "content": "Translate to French: 'Hello'"}])
print(response["choices"][0]["message"]["content"])
```

--- 


## Custom Callbacks

Callbacks allow you to track API calls, measure latency, log failures, or modify requests before sending them. This is useful for monitoring API usage in production applications.

This is how to add custom callbacks to LiteLLM. 

```python

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm import completion

import logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filemode='w',
                    filename='litellm.log')
logger = logging.getLogger(__name__)


# Custom Logger 
class MyCustomHandler(CustomLogger):
    def log_pre_api_call(self, model, messages, kwargs): 
        logger.info(f"Pre-API Call")
      ...

# initialize the handler
customHandler = MyCustomHandler()
# pass the handler to the callback
litellm.callbacks = [customHandler]

response = completion(
    model="ollama/llama3.2",
    messages=[{ "content": "Write some python code to print the contents of a file.","role": "user"}],
    stream=False
)
print(response.choices[0].message.content)


```

In the log you will see
```
2025-02-24 23:18:32,187 - __main__ - INFO - Pre-API Call
```


### Callback Functions

If you just want to log on a specific event (e.g. on input) - you can use callback functions.

You can set custom callbacks to trigger for:

litellm.input_callback - Track inputs/transformed inputs before making the LLM API call
litellm.success_callback - Track inputs/outputs after making LLM API call
litellm.failure_callback - Track inputs/outputs + exceptions for litellm calls


```python
def custom_callback(
    kwargs,                 # kwargs to completion
    completion_response,    # response from completion
    start_time, end_time    # start/end time
):
    # Your custom code here
    print("LITELLM: in custom callback function")
    print("kwargs", kwargs)
    print("completion_response", completion_response)
    print("start_time", start_time)
    print("end_time", end_time)


import litellm
litellm.success_callback = [custom_callback]
```

---

## Use Cases

### 1. **Building AI-Powered Chatbots**
Easily integrate LLMs into chatbot applications with failover mechanisms to ensure reliability.

### 2. **Cost-Optimized AI Applications**
Use a mix of free and paid models, switching dynamically based on cost and performance needs.

### 3. **Enterprise AI Deployment**
Organizations can route queries across different LLM providers to ensure uptime and efficiency.

### 4. **Research and Development**
Experiment with various LLMs without rewriting API calls for each provider.

---


LiteLLM provides an elegant way to simplify LLM interactions, reduce API complexity, and optimize costs. Another key point is it is supported and supports a large amount of the current LLM providers.