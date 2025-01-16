+++
date = '2025-01-07T16:51:43Z'
draft = true
title = 'Agents: A tutorial on building agents in python'
categories = ['AI', 'agents', 'smolagents']
tag = ['agents'] 
+++

## LLM Agents

Agents are used enhance and extend the functionality of LLM's.

In this tutorial, we’ll explore what LLM agents are, how they work, and how to implement them in Python. 

## What Are LLM Agents?

An agent is an autonomous process that may use the LLM and other tools multiple times to achieve a goal. 
The LLM output often controls the workflow of the agent(s).

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

Agents can be used anywhere, some common applications include

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

### 3. Create an agent that will call a local ollama model

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

The result is:

```
1 import random                         
2                                       
3 def print_random_numbers(count):      
4     for _ in range(count):            
5         print(random.randint(1, 100)) 
6                                       
7 print_random_numbers(100)             

Execution logs:
55
79
94
93
...

```



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
New York City-based NGO Human Rights Watch, in a report entitled Laws of Attrition, authored by Hugh Williamson, the British director of HRW's Europe & Central Asia Division, has claimed that since May 2012, when Putin was reelected as president, Russia has enacted many restrictive laws, started inspections of non-governmental organizations  
...

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

### 6. Combining multiple tools to get a job done

In this example I am using multiple tools to achive a task.

To define a new tool Use the @tool 
We define two tools 

- **send_request** to get the contents of a web page
- **write_to_file** to save the data to a file

In this application we take the results of running these tools on an website page and then use the model to summarize that result.


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


### 7. Multi Agents

If we have hundreds of agents working together over a large network maybe this can scale the way corporations and human effort does.
If LLM's start to slow down well then we can use this approach to achieve  greater goals.
This is the aim of using multiple agents to solve problems.

```python

import re
import requests
from markdownify import markdownify
from requests.exceptions import RequestException
from smolagents import tool


@tool
def visit_webpage(url: str) -> str:
    """Visits a webpage at the given URL and returns its content as a markdown string.

    Args:
        url: The URL of the webpage to visit.

    Returns:
        The content of the webpage converted to Markdown, or an error message if the request fails.
    """
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Convert the HTML content to Markdown
        markdown_content = markdownify(response.text).strip()

        # Remove multiple line breaks
        markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)

        return markdown_content

    except RequestException as e:
        return f"Error fetching the webpage: {str(e)}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"


print(visit_webpage("https://en.wikipedia.org/wiki/Hugging_Face")[:500])

```

The result here is:

```
Hugging Face - Wikipedia

[Jump to content](#bodyContent)

Main menu
...
```

We now use this tool to build our multi agent system

```Python
from smolagents import (
    CodeAgent,
    ToolCallingAgent,
    HfApiModel,
    ManagedAgent,
    DuckDuckGoSearchTool,
    LiteLLMModel,
)

model = LiteLLMModel(
    model_id="ollama/qwen2.5:latest",
    api_base="http://localhost:11434"
)

web_agent = ToolCallingAgent(
    tools=[DuckDuckGoSearchTool(), visit_webpage],
    model=model,
    max_steps=10,
)

# wrap this agent into a manged agent
managed_web_agent = ManagedAgent(
    agent=web_agent,
    name="search",
    description="Runs web searches for you. Give it your query as an argument.",
)

```

Finally we create a manager agent and pass our managed agent to it in its managed_agents argument  

```Python
manager_agent = CodeAgent(
    tools=[],
    model=model,
    managed_agents=[managed_web_agent],
    additional_authorized_imports=["time", "numpy", "pandas"],
)
```

Lets test it out:

```Python
answer = manager_agent.run("When did Ireland gain independence from Britain")
```


```
Final answer: ### 1. Task outcome (short version):
Ireland gained independence from Britain in stages through a process involving both legislative and military efforts.

### 2. Task outcome (extremely detailed version):
The process of Ireland gaining independence from Britain involved several key events and phases:

- **Home Rule Act 1914**: This act was intended to grant Home Rule to Ireland, allowing it greater self-governance within the United Kingdom. However, its implementation was delayed due to the 
outbreak of World War I.
  
- **Anglo-Irish Treaty 1921**: Signed in December 1921 between representatives of the British government and Sinn Féin, this treaty established an Irish Free State (later known as Ireland) under the 
Crown. It granted significant self-governance to most of Ireland while Northern Ireland opted out.
  
- **Irish Independence Act 1922**: Enacted in July 1922, this act formally separated the Irish Free State from the United Kingdom, though it retained certain ties with Britain.

- **Proclamation of the Republic (1948)**: In April 1948, Ireland officially declared itself a republic and withdrew its remaining connection to the British Crown, effectively completing its 
independence process. The Republic of Ireland Act was passed on April 18, 1948.

### 3. Additional context:
The transition from British rule to full Irish independence involved complex political dynamics, including the Irish War of Independence (1919-1921) and subsequent civil war in Ireland (1922). The 
Treaty of 1921 established a form of shared sovereignty between the Crown and the Free State, which was later fully resolved by declaring a republic.

### Additional Prompting:
For a deeper understanding, consider exploring the specific events leading up to each agreement or treaty, the roles of key political figures like Éamon de Valera and Winston Churchill, and the social
and economic conditions that influenced Ireland's pursuit of independence.

```

## Best Practices for Building LLM Agents

- **Version your changes**: You need a rigid structure for making changes to your application. Some form of [Gitflow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow). If you do not have a consistent development and change tracking process you could find that you loose literally man years of work.
- **Optimize Prompts**: Experiment with prompt phrasing to improve accuracy.
- **Use open models**: 
    - Where possible use Ollama. 
    - Its free and for most tasks it will be competitive with the paid models. 
    - You can offload some of the work or some of the agents to Ollama. 
    - Always design your application so that this can be configured on/off.
    - Split your functionality into discrete tasks it wil allow for mixing and matching between the open and paid versions.
    - There are always new open models being release that often are better than the psid versions at specific tasks. 
- **Track your expenses**: If using a paid model make it really obvious what each call or agent is costing. I suggest adding a report with a per service/agent price generated every hour with a graph and comparison over time.
- **Limit Token Usage**: Use concise prompts and responses to manage costs.
- **Use Fine-Tuning**: For domain-specific tasks, fine-tune the model on your data.
- **Implement Logging**: Log inputs, outputs, and errors for debugging and analysis.
- **Configure and track**: Make everything models used, agents, memory server a config option and dump this configuration at the beginning of each run. When things go astray you can compare teh configs to see what has gone has changed.
- **Secure APIs**: Protect your API keys and implement rate limiting.
- **Escape the Jupyter Labyrinth**: Jupyter notebooks are awesome they allow researchers engineers and managers to collaborate and work together on ideas and immediately see and share the results. This can lead to a massive spaghetti solution comprising lots of notebooks. You want to avoid this.
---


### 8. Advanced applications of Agents

AI Agents are currently the **state of the art** SOTA in LLM development.
If we do not get massive improvements in LLMs this year by using agents we may be able to achieve a similar outcome.

These papers show how agents can improve teh performance of LLMs

1. **Self-Improvement in Web Agent Tasks**  
   - Research showing a **31% increase** in task completion rates on the WebArena benchmark through self-improvement:  
   [Read the paper](https://arxiv.org/abs/2405.20309)

2. **Dynamic LLM-Agent Networks (DyLAN)**  
   - Achieved a **13.0% improvement** on the MATH dataset, a **13.3% improvement** on the HumanEval benchmark, and up to a **25.0% increase** on specific MMLU tasks:  
   [Read the paper](https://arxiv.org/abs/2310.02170)

3. **Enhanced Resume Screening with LLM Agents**  
   - Demonstrated an **11x faster process** and significant improvements in performance metrics for automated resume screening:  
   [Read the paper](https://arxiv.org/abs/2401.08315)

---


## Real-World Applications

LLM agents can power:

- **Chatbots**: Automate customer support with contextual responses.
- **Data Analysis Assistants**: Query databases or analyze datasets via natural language.
- **Code Review Tools**: Provide feedback on code snippets or suggest improvements.
- **Autonomous Agents**: Perform multi-step workflows, such as booking appointments or managing emails.
- **YouTube Filter Tools**: You build an agents to filter the content to just what you are interested in and then summarize those videos ot save time. 
- **Job Search Tools**: You can build agents to automate the process of looking for jobs. 
- **Personalized Recommendations**: You can track yourself and what you like and would want to buy privately, look out for bargains and get notified of the best deals at the bes time to buy.

---

## Resources

-[smolagents](https://huggingface.co/docs/smolagents/index)
This is a new library from [huggingface](https://huggingface.co/). It is where I would start building agents. Its new, works with local models supports running code.

-[langraph](https://github.com/langchain-ai/langgraph)
These guys have a massive amount of tools with lots of tutorials. I will do another blog post on these guys.

-[llamaindex](https://github.com/run-llama/llama_index)
More focussed on data management. They have lots of cool tools and tutorial online.

## Code examples

You can find example notebooks here:

[Agents](https://github.com/ernanhughes/agents)