+++
date = '2025-01-10T20:32:03Z'
draft = true
title = 'CAG: Cache-Augmented Generation'
+++

## Summary

**CAG performs better but does not solve the key reason RAG was created small context windows.**

Retrieval-Augmented Generation (RAG) is currently(early 2025) the most popular way to use external knowledge in current LLM opperations. RAG allows you to enhance your LLM with data beyond the data it was trained on. Ther are many great RAG solutions and products. 

RAG has some drawbacks 
    - There can be significant retreival latency as it searches for and organizes the correct data.  
    - There can be errors in the documents/data it selects as results for a query. For example it may select the wrong document or give priority to the wrong document.
    - It may interduce security and data issues [2️⃣](#references).  
    - It introduces complication  
        - an external application to manage the data (Vector Database)
        - a process to continually update this data when the data goes stale
     

Cache-Augmented Generation (CAG) proposed as an alterative approach [1️⃣](#references) I suggest could make a great complement for RAG.
    - Is faster then RAG
    - Is useful where the data size can fit into the context window of your model. 
    - Is less complicated than RAG
    - Can give better quality results than RAG (BERTScore)


I am going to cover CAG in this post.

## What is CAG

CAG prepocess all relavent data and loads this data in the LLM's extended context windows.

- **Prepare Dataset**: Preprocess all relevant knowledge documents.  
- **Preload Context**: Load the dataset into the LLM’s extended context window. For example if using Ollama the default context is 2048 however many models suppost up to 128K context. When using the model for CAG you will need to configure the model to use it full context (or as much as you need).  
- **Cache Inference State**: Store the inference state for repeated queries.  
    Query Model: Directly interact with the model using the cached knowledge.  
    Generate Outputs: Produce final results without retrieval latency.  
- **Cache Reset**: After generating a response the KVCache can be reset to its inital state to prepare for the next query.

## What are the benefits and drabacks of CAG
It is faster than RAG and less complicated however it does require a large context size as it is populated with all the documentation. CAG's primary limitation is the context window size as this will cap the amount of preloaded data.


|Feature| **RAG** | **CAG** |  
| ----- | ----- | ----- |  
|Performance|Performs real-time retrieval of information during inference. This can slow down response times.|Preloads all relevant knowledge into the model’s context, providing faster response times.|  
|Errors|Subject to potential errors in document selection and ranking.|Minimizes data errors by ensuring holistic context is present.|  
|Complexity|Integrates retrieval, update and generation components, which increases system complexity.|Simplifies architecture however we do need to maange the KVCache.|  
|Context|Dynamically added with each new query.|Context from preloaded data.|  
|Memory Usage|Uses additional memory and resources for external retrieval.|Uses preloaded KV-cache for efficient resource management. However larger context can lead to challenges.|
|Correctness|Is a standard solution usually built using fit for puropose tools like postgres.|Is an unsupported solution using the cache to store non transient data which it was not designed to do.|


## When should you use CAG

Datasets that don’t change frequently.
Small to Medium Dataset Size: Knowledge fits within the LLM’s context window (32k–100k tokens).
Low-Latency Use Cases: Scenarios where speed is critical (e.g., real-time chat systems).

## Calculating the number of tokens in text

[llama-token-counter](https://huggingface.co/spaces/Xanthius/llama-token-counter) can be used to calculate tokens.

To give you an idea of tokens I used it to calculate token count for the bible.

```
Authorized (King James) Version (AKJV) 
Characters (without line endings): 4,478,421
Words: 890,227
Lines: 31,105
Document length: 4,544,631

Number of tokens: 1365259
```

### Code to calculate token count

```Python
def get_token_count(text, model_name="meta-llama/Meta-Llama-3-8B"):
  """
  Calculates the number of tokens in the given text for a specified LLaMA model.

  Args:
    text: The input text string.
    model_name: The name of the LLaMA model (default: "meta-llama/Meta-Llama-3-8B").

  Returns:
    The number of tokens in the text.
  """
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  tokens = tokenizer.encode(text)
  return len(tokens)

def get_text_from_file(file_path):
  """
  Reads the entire contents of a text file.

  Args:
    file_path: The path to the text file.

  Returns:
    The contents of the file as a single string.
  """
  try:
    with open(file_path, 'r') as file:
      text_data = file.read()
    return text_data
  except FileNotFoundError:
    print(f"Error: File not found at '{file_path}'")
    return None


data_path = os.path.abspath("shakespeare.txt")
text_data = get_text_from_file(data_path)
token_count = get_token_count(text_data)
print(f'Number of tokens in the file {data_path}: {"{:,}".format(token_count)}')

```

## Determine the usable context size for your model

[RULER: What’s the Real Context Size of Your Long-Context Language Models?](https://github.com/NVIDIA/RULER)  some research from NVIDA which shows an effective lenght for many open source models.

Essentially what they found was as the context lenght increases the model perfromance decreases.

The key point I would take from this research is that even though your model may claim to have a 128K context it may really only support half this amount with acceptable quality.




## CAG Implementation

### Generate the kv for the model

```Python
def generate(model, input_ids: torch.Tensor, past_key_values, max_new_tokens: int = 50) -> torch.Tensor:
    """
    Generates a sequence of tokens using the given model.
    Args:
        model: The language model to use for generation.
        input_ids (torch.Tensor): The input token IDs.
        past_key_values: The past key values for the model's attention mechanism.
        max_new_tokens (int, optional): The maximum number of new tokens to generate. Defaults to 50.
    Returns:
        torch.Tensor: The generated token IDs, excluding the input tokens.
    """
    device = model.model.embed_tokens.weight.device
    origin_len = input_ids.shape[-1]
    input_ids = input_ids.to(device)
    output_ids = input_ids.clone()
    next_token = input_ids

    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(input_ids=next_token, past_key_values=past_key_values, use_cache=True)
            logits = out.logits[:, -1, :]
            token = torch.argmax(logits, dim=-1, keepdim=True)
            output_ids = torch.cat([output_ids, token], dim=-1)
            past_key_values = out.past_key_values
            next_token = token.to(device)

            if model.config.eos_token_id is not None and token.item() == model.config.eos_token_id:
                break

    return output_ids[:, origin_len:]


def get_kv_cache(model, tokenizer, prompt: str) -> DynamicCache:
    """
    Generates a key-value cache for a given model and prompt.
    Args:
        model: The language model to use for generating the cache.
        tokenizer: The tokenizer associated with the model.
        prompt (str): The input prompt for which the cache is generated.
    Returns:
        DynamicCache: The generated key-value cache.
    """
    device = model.model.embed_tokens.weight.device
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    cache = DynamicCache()

    with torch.no_grad():
        _ = model(input_ids=input_ids, past_key_values=cache, use_cache=True)
    return cache


def clean_up(cache: DynamicCache, origin_len: int):
    """
    Trims the key_cache and value_cache tensors in the given DynamicCache object.

    Args:
        cache (DynamicCache): The cache object containing key_cache and value_cache tensors.
        origin_len (int): The length to which the tensors should be trimmed.

    Returns:
        None
    """
    for i in range(len(cache.key_cache)):
        cache.key_cache[i] = cache.key_cache[i][:, :, :origin_len, :]
        cache.value_cache[i] = cache.value_cache[i][:, :, :origin_len, :]
```

### 

### How to use the CAG

```Python

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# load a model, here I chose phi because it is small with a big context
model_name = "microsoft/Phi-3-mini-128k-instruct"
hf_token = os.getenv("HF_TOKEN")
tokenizer = AutoTokenizer.from_pretrained(
    model_name, token=hf_token, trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True,
    token=hf_token,
)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Loaded {model_name}.")

#load a file for testing
with open("genesis.txt", "r", encoding="utf-8") as f:
    doc_text = f.read()

system_prompt = f"""
<|system|>
You are an assistant who provides concise factual answers.
You strive to just answer the user's question.
<|user|>
Context:
{doc_text}
Question:
""".strip()

genesis_cache = get_kv_cache(model, tokenizer, system_prompt)
origin_len = genesis_cache.key_cache[0].shape[-2]
print("KV cache built.")


# query the cache
question1 = "Why did God create eve?"
clean_up(genesis_cache, origin_len)
input_ids_q1 = tokenizer(question1 + "\n", return_tensors="pt").input_ids.to(device)
gen_ids_q1 = generate(model, input_ids_q1, genesis_cache)
answer1 = tokenizer.decode(gen_ids_q1[0], skip_special_tokens=True)
print("Q1:", question1)
print("A1:", answer1)


```


## Practical Implementations

The reason we have RAG is we do not have large enough contexts to handle our data requests. CAG does not solve this problem RAG does. 
CAG does not support dynamic data arguably RAG does not either but it has solutions and approaches for this.
Your puttting data into an unmanaged memory area (essentially the cache is not designed for this). Compare it to RAG which typicall backends to an [ACID](https://en.wikipedia.org/wiki/ACID) database.
LLMs particularilly closed ones may not provide a hook where you can inject your KV structures into the process. 


If you have an application the manages a model and you need max performance over a dataset that is relatively static this is a solution. For example you are building a personal chatbot with a users diary or lifetime conversaions with the bot.


## References

1️⃣ [Don't Do RAG: When Cache-Augmented Generation is All You Need for Knowledge Tasks](https://arxiv.org/pdf/2412.15605v1.pdf)

2️⃣ [Pirates of the RAG: Adaptively Attacking LLMs to Leak Knowledge Bases](https://arxiv.org/abs/2412.18295)  

3️⃣ [Not All Contexts Are Equal: Teaching LLMs Credibility-aware Generation](https://arxiv.org/pdf/2404.06809)

[RULER: What's the Real Context Size of Your Long-Context Language Models?](https://arxiv.org/abs/2404.06654)

## Code examples

[CAG notebooks](https://github.com/ernanhughes/cag-notebooks)
https://github.com/ernanhughes/cag-noteboooks

Has some example code for using CAG.
