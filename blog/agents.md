+++
date = '2025-01-07T16:51:43Z'
draft = true
title = 'Agents'
categories = ['AI', 'agents', 'smolagents']
tag = ['agents'] 
+++


# LLM Agents

Agents are used enhance and extend the functionality of LLM's.

In this tutorial, we’ll explore what LLM agents are, how they work, and how to implement them in Python. 
---

## What Are LLM Agents?

An agent is an autonomous process that may use the LLM and other tools multiple times to achieve a goal. 
The LLM output often controls the workflow of the agents.

## What is the difference between Agents and LLMs or AI?

Agents are processes that may use LLM's and other agents to achieve a task. 
Agents act as orchestrators or facilitators, combining various tools and logic, whereas LLMs are the underlying generative engines.

For example a customer support agent could be comprised of 
- an agent to retrieve customer information 
- an agent to retrieve order details and information
- a processing agent to cancel or update orders
- a chatbot to interact with the customer
- a guardrails agent to make sure that the customer has an optimal experience
- a human in the loop agent to forward the customer to a real person when no solution is achieved


### Sample applications of Agents

Agents can be used anywhere some common applications include

- Process unstructured natural language input.
- Perform reasoning to deduce solutions or actions.
- Interact with external systems, APIs, or databases.
- Generate human-like responses or results based on context.
- Add a memory to the LLM.
- Query the web for contextual information.
- Run python code to extract further information from the data available to the LLM


### Autonomous operation

They are often designed to act autonomously, making them useful for automating workflows, building chatbots, or even creating virtual assistants.


### Agent collaboration

Ofter we have agents interacting with other agents and in some cases managing and controlling other agents to perform a task.

### Designing Agent solutions

When designing these solutions It can be helpful to think of the process as composed of different sections.

- If we think the core agent as a **manager**.
- This manager has a **goal** to accomplish. 
- To achieve this goal he has a **process** to complete.
- To do this process he needs a certain **group of individuals**.
- He needs to determine **which individuals (agents) are best suited** to achieve this goal 
- Each of these individuals is then given a **task**.
- The progress is **monitored, measured and evaluated** constantly.


---

## How Do LLM Agents Work?

At a high level, an LLM agent typically involves:

1. **Prompt Engineering**: Crafting inputs to the language model to achieve desired outputs.
2. **Memory Management**: Storing and retrieving relevant context to maintain coherence.
3. **Tool Integration**: Connecting the agent to APIs, databases, or external tools for enhanced functionality.
4. **Decision Logic**: Implementing policies or scripts to guide the agent’s behavior.

---

## Setting Up LLM Agents with Python

Let’s walk through how to build an LLM agent using Python. 
We’ll use the [smolagents](https://github.com/huggingface/smolagents) library for this example.

### 1. Install Necessary Libraries

First, install the required Python packages:

```bash
pip install smolagents, accelerate
```

### 2. Create a simple AI Agent

This agent uses the DuckDuckGoSearchTool and a Hugging face model to estimate the time required for a horse to run the length of O'Connell Street.

```python
from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel

agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=HfApiModel())

agent.run("How many seconds would it take for a horse running at full speed to run the length of O'Connell Street in Dublin?")

```

This will generate a lot of details about the steps performed 

```
1. length_query = "Length of O'Connell Street Dublin
2. length_results = web_search(query=length_query) 
print(length_results)

1 speed_query = "Top speed of a horse"
2 speed_results = web_search(query=speed_query)
3 print(speed_results)


 # Constants 
 length_of_street_meters = 500  # Length of O'Connell Street in meters 
 horse_speed_mph = 44  # Top speed of a horse in miles per hour  
 # Conversion factors  │
 miles_to_meters = 1609.34  # 1 mile = 1609. 34 meters 
 hours_to_seconds = 3600  # 1 hour = 3600 seconds  
 # Convert speed from mph to m/s 
 horse_speed_mps = horse_speed_mph * miles_to_meters / hours_to_seconds  
 # Calculate time in seconds 
 time_seconds = length_of_street_meters / horse_speed_mps  
 final_answer(time_seconds)  

```
and then generate a result:

```
Out - Final answer: 25.41979377203755

```

### 3. create an agent that will call a local ollama model

In this case I wan to call a local ollama model to generate some python code.

Note make sure you have the required model installed.

```
ollama pull qwen2.5-coder:14b:latest
```


```python

from smolagents import CodeAgent, LiteLLMModel

model_id = "ollama/qwen2.5-coder:14b"

model = LiteLLMModel(model_id=model_id, api_base="http://localhost:11434")
agent = CodeAgent(tools=[], model=model, add_base_tools=True)

agent.run(
    "Can you write a python function to print 100 random numbers?",
)

```
1. To call local ollama we use the LiteLMModel class
2. In your local ollama installation you can list the models by calling ollama list
3. We need to pas in the api_base url in this case it is on my local machine.


### 4. Simple web search tool

This does a simple web search

```Python

from smolagents import load_tool

search_tool = load_tool("web_search")
print(search_tool("Who's the current president of Russia?"))

```

This will return this result.

```
## Search Results

...
[Vladimir Putin - Wikipedia](https://en.wikipedia.org/wiki/Vladimir_Putin)
New York City-based NGO Human Rights Watch, in a report entitled Laws of Attrition, authored by Hugh Williamson, the British director of HRW's Europe & Central Asia Division, has claimed that since May 2012, when Putin was reelected as president, Russia has enacted many restrictive laws, started inspections of non-governmental organizations ...

```


### 5. Add Tool Integration

LLM agents become more powerful when they interact with external tools. For example, integrating a search API:

```python
import requests

def search_tool(query):
    api_url = f"https://api.example.com/search?q={query}"
    response = requests.get(api_url)
    return response.json()

# Agent combining LLM and search
def agent_with_tool(prompt):
    if "search" in prompt:
        query = prompt.split("search")[-1].strip()
        search_results = search_tool(query)
        return f"Search Results: {search_results}"
    else:
        return simple_llm_agent(prompt)

print(agent_with_tool("search Python decorators"))
```

### Combining tools

In this example I am using multiple tools to achive a task.

To define a new tool Use the @tool 
We define two tools 

- **send_request** to get the contents of a web page
- **write_to_file** to save the data to a file

In this application we want to use the model to summarize an article from a website.


```Python
from typing import Optional
from smolagents import CodeAgent, tool, LiteLLMModel, ToolCallingAgent

import urllib.request
from markdownify import markdownify as md

@tool
def send_request(addr: str) -> str:
    """Sends http get request to the given address and returns the response as markdown
    
    Args:
        addr: The address to send the request to
    """

    return md(urllib.request.urlopen(addr).read())

@tool
def write_to_file(name: str, content: str) -> Exception:
    """Write the given content to the file with given name
    
    Args:
        name: The name of the file
        content: The content of the file as a string
    """
    #create file if not exists and handle already exist error
    with open(name, "wb") as f:
        out = f.write(content)
        if out == 0:
            raise Exception("Error writing to file")
    return None

model = LiteLLMModel(
    model_id="ollama/qwen2.5:latest",
    api_base="http://localhost:11434"
)

agent = CodeAgent(tools=[send_request, write_to_file], model=model, additional_authorized_imports=["datetime", "bytearray"])

print(agent.run("Summarize this website page http://programmer.ie/post/ollama/"))

```

The result is 

```
Executing this code: 
1 response = send_request(addr='http://programmer.ie/post/ollama/)
2 webpage_content = response.text 
4 # Print the webpage content for inspection. In a real scenario, we would analyze this content.
5 print webpage_content)
6
7 # For now, let's assume an example summary based on the page title and some text snippets
8 summary = "The post discusses the Ollama project, which is related to machine learning and AI models.
9 final_answer(summary)

```




### 5. Implement Memory Management

To maintain context, you can add memory to your agent. A simple way is to use a list to store conversation history:

```python
conversation_history = []

def agent_with_memory(prompt):
    global conversation_history
    conversation_history.append(prompt)
    context = "\n".join(conversation_history)
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=context,
        max_tokens=150
    )
    reply = response.choices[0].text.strip()
    conversation_history.append(reply)
    return reply

print(agent_with_memory("What is Python?")
print(agent_with_memory("How does it handle memory?"))
```

### 6. Advanced Features with LangChain

LangChain simplifies building LLM-powered applications. Here’s an example of chaining prompts and tools:

```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

llm = OpenAI(model_name="text-davinci-003")
prompt = PromptTemplate(template="Answer the following: {question}", input_variables=["question"])
chain = LLMChain(llm=llm, prompt=prompt)

print(chain.run("Explain Python generators"))
```

---

## Best Practices for Building LLM Agents

- **Optimize Prompts**: Experiment with prompt phrasing to improve accuracy.
- **Limit Token Usage**: Use concise prompts and responses to manage costs.
- **Use Fine-Tuning**: For domain-specific tasks, fine-tune the model on your data.
- **Implement Logging**: Log inputs, outputs, and errors for debugging and analysis.
- **Secure APIs**: Protect your API keys and implement rate limiting.

---

## Real-World Applications

LLM agents can power:

- **Chatbots**: Automate customer support with contextual responses.
- **Data Analysis Assistants**: Query databases or analyze datasets via natural language.
- **Code Review Tools**: Provide feedback on code snippets or suggest improvements.
- **Autonomous Agents**: Perform multi-step workflows, such as booking appointments or managing emails.

---

## Conclusion

LLM agents offer unparalleled flexibility and intelligence, enabling Python developers to build smarter, more interactive applications. By leveraging tools like OpenAI, LangChain, or LlamaIndex, you can craft agents that transform user interactions into dynamic, context-aware experiences.

Ready to start building? Explore the documentation for [OpenAI](https://platform.openai.com/docs/), [LangChain](https://docs.langchain.com/), and [LlamaIndex](https://gpt-index.readthedocs.io/) to unlock the full potential of LLM agents in your projects.

