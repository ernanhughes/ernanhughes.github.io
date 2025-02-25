+++
date = '2025-02-23T21:47:10Z'
draft = false
title = 'TextGrad: dynamic optimization of your LLM'
categories = ['TextGrad']
tag = ['textgrad'] 
+++

## **Summary** 

This post aims to be a comprehensive tutorial on [Textgrad](https://github.com/zou-group/textgrad).

Textgrad enables the optimization of LLM's using their text responses.

This will be part of `SmartAnswer` the ultimate LLM query tool which I will be blogging about shortly.

---

## Why TextGrad?

* Brings Gradient Descent to LLMs – Instead of numerical gradients, TextGrad leverages textual feedback to iteratively improve outputs.
* Automates Prompt Optimization – Eliminates the guesswork in refining LLM prompts.
* Works with Any LLM – From OpenAI’s GPT to local models like Ollama.

---

## **What is TextGrad?**
### **Bringing Gradients to LLM Optimization**
Traditional AI optimization techniques rely on numerical gradients computed via backpropagation. However in **LLM-driven AI systems**, inputs and outputs are often text, making standard gradient computation impossible.

**TextGrad** introduces a powerful metaphor: *textual gradients* natural language feedback that serves as an analogue to numerical gradients. Instead of computing numerical derivatives, TextGrad uses **LLMs to generate structured feedback** that guides optimization. This feedback is then propagated backward to refine prompts, solutions, or even the behavior of an entire AI system.

> **Key Insight**: TextGrad enables **automatic differentiation via text**, making it possible to optimize AI-generated outputs without requiring explicit mathematical gradients.

## **How It Works**
The TextGrad framework follows an optimization process similar to PyTorch:


### **How Traditional Gradient-Based Optimization Works**
In traditional **gradient-based optimization**, a model learns by computing **gradients** mathematical derivatives that tell it how to adjust its parameters to minimize a loss function. This is done using **backpropagation**, where the error is propagated backward through the network.

For example, in neural networks:
1. **Forward pass**: The model makes a prediction.
2. **Loss computation**: A function measures how far the prediction is from the expected result.
3. **Backward pass**: The loss is differentiated to compute gradients, which indicate how to change model parameters.
4. **Parameter update**: Using an optimizer like **SGD** or **Adam**, the model updates itself.

---

### **How TextGrad Uses LLMs Instead of Numerical Gradients**
However, **LLM-driven systems** don't have parameters that can be directly adjusted via numerical gradients. Instead of mathematical differentiation, **TextGrad** uses **structured textual feedback** as an alternative to gradients.

### **Step-by-Step Process of TextGrad's Optimization**
1. **Forward Pass: Generate an Output**  
   - An LLM is given an **input prompt** and generates an initial response.
   - Example: *"What is the capital of France?"* → LLM says *"Paris is the capital of France."*

2. **Loss Computation: Evaluate the Output Using Textual Feedback**  
   - Instead of computing a mathematical loss function, TextGrad asks an **evaluator LLM** to critique the response.
   - Example: *"Evaluate if the response correctly answers the question."*
   - The LLM might generate feedback:  
     - *"The response is correct, but lacks additional context about France."*
     - *"Consider mentioning that Paris is also the largest city."*

3. **Backward Pass: Interpret the Feedback as a Gradient**  
   - TextGrad interprets this **LLM-generated feedback** as **textual gradients**.
   - These textual gradients describe **how the response should be changed** to improve.

4. **Optimization Step: Adjust the Input and Re-run the LLM**  
   - The input is modified **based on the feedback** and re-run through the LLM.
   - Example:
     - Original prompt: *"What is the capital of France?"*
     - Optimized prompt: *"Explain the capital of France, including historical and geographical details."*
   - The LLM now generates a **more refined answer**.


--- 


## How to use local models with textgrad

To enable local models use `experimental` as a prefix to the LiteLLM model name.

```python
import textgrad as tg
from textgrad import get_engine, set_backward_engine
import litellm

# I like this it really helps with debugging if you see too many messages turn comment it out
litellm._turn_on_debug()

# this will use litellm to call your local Ollama model
MODEL_NAME = "ollama/Qwen2.5"

# engine is the model
engine = get_engine(f"experimental:{MODEL_NAME}", cache=False)
set_backward_engine(f"experimental:{MODEL_NAME}", cache=False, override=True)

```


---

## TextGrad Optimization Process

### Step 1: Define the Variables (Input)

Mainly these are prompts that you are optimizing or are using to enhance other processes.

* **role_description** the role of this variable
* **value** the actual thing we are after from the llm.
* **predecessors** predecessors of this variable in the computation graph
* **requires_grad** whether this variable requires a gradient, defaults to True.

```python
MODEL_NAME = "ollama/Qwen2.5"

engine = get_engine(f"experimental:{MODEL_NAME}", cache=False)
evaluation_instruction = Variable("Is this a good joke?", 
                                  role_description="question to the LLM",
                                  requires_grad=False)
response_evaluator = TextLoss(evaluation_instruction, engine)
response
```
```
Variable(value=That's an interesting setup! It’s quite minimalist and leaves the audience to fill in the gaps with their imagination. However, it might benefit from a punchline or a twist to make it more of a joke. For example:

"A blind man walks into a bar. And a table. And a chair. Then he orders two beers because he can feel them."

This adds a bit of humor and surprise, making the situation funnier and more engaging for the listener., role=response from the language model, grads=set())
```

---

### Step 2: Set up the Engine

The engine is an abstraction used to interact with the model

```python
MODEL_NAME = "ollama/Qwen2.5"

engine = get_engine(f"experimental:{MODEL_NAME}", cache=False)
# this also works with
set_backward_engine(f"experimental:{MODEL_NAME}", cache=False, override=True)
engine.generate("Hello how are you?")
```
```
"Hello! I'm here and ready to assist you. How can I help you today?"
```

---

### Step 3: Define the Loss Function

This has a similar concept to loss in pytorch.

```python
system_prompt = Variable("Evaluate the correctness of this sentence", role_description="The system prompt")
loss = TextLoss(system_prompt, engine=engine)
loss
```

#### The general steps for loss function design in TextGrad 

* **1. Define the Objective**: Determine the goal you want to achieve, whether it's improving code, answering questions, optimizing molecular properties, or treatment plans.
* **2. Create a Computation Graph**: Represent the AI system as a computation graph, where variables are inputs and outputs of complex function calls.
* **3. Specify the Loss Function**: Define the loss function in a way that aligns with the objective. This could involve using an LLM to evaluate the output, running unit tests on code, or using simulation engines to assess molecular properties.
* **4. Implement Backpropagation**: Use TextGrad's backpropagation algorithm to compute gradients and provide feedback on how to improve the variables.
* **5. Apply Textual Gradient Descent (TGD)**: Use TGD to update the variables based on the gradients, iteratively refining the system.

---

### Step 4: Perform Optimization

The optimizer in TextGrad is the object that will update the parameters of the model. In this case, the parameters are the variables that have `requires_grad` set to True.

This example shows how we can fix an input using `textgrad`.

```python
from textgrad import Variable, TextLoss, TextualGradientDescent
MODEL_NAME = "ollama/llama3.2"
x = Variable("Tre is somtin about this sentance tha tis not quite right.", role_description="The input sentence", requires_grad=True)
engine = get_engine(f"experimental:{MODEL_NAME}", cache=False)
# this also works with
set_backward_engine(f"experimental:{MODEL_NAME}", cache=False, override=True)
x.gradients
system_prompt = Variable("Evaluate the correctness of this sentence", role_description="The system prompt")
loss = TextLoss(system_prompt, engine=engine)
optimizer = TextualGradientDescent(parameters=[x], engine=engine)
l = loss(x)
l.backward()
optimizer.step()
x.value

```
```
"The is something about this sentence that's not quite right."
```

---

## Use cases of TextGrad

### **Example 01: Improving LLM Responses**

Let’s say we want to optimize an LLM’s reasoning abilities in a math problem:

```python
import textgrad as tg

# Define a math problem
question = tg.Variable("What is the sum of the first 100 positive integers?", 
                       role_description="question for llm",
                       requires_grad=False)

engine = get_engine(f"experimental:{MODEL_NAME}", cache=False)
# this also works with
set_backward_engine(f"experimental:{MODEL_NAME}", cache=False, override=True)

# Get the initial response
model = tg.BlackboxLLM(engine=engine)
answer = model(question)

# Define textual feedback as a loss function
evaluation_instruction = "Evaluate if the response follows a correct step-by-step approach."
loss_fn = tg.TextLoss(evaluation_instruction)

# Compute the loss and optimize
loss = loss_fn(answer)
loss.backward()
optimizer = tg.TGD(parameters=[answer])
optimizer.step()

# Print the refined response
print("Optimized Answer:", answer.value)
```


```
Optimized Answer: The sum of the first n positive integers can be calculated using the formula: `n(n+1)/2`. This method has been used for centuries, dating back to ancient civilizations such as the Babylonians and Greeks. When `n` is very large, the formula can become computationally intensive due to the multiplication operations involved. However, most modern computers have optimized algorithms for handling such cases efficiently.

For example, let's consider the case where `n = 100`, which is a relatively small value compared to some of the largest numbers used in mathematics and engineering. In this scenario, the formula provides an accurate result quickly without requiring extensive computational resources.

It's also worth mentioning that this formula assumes a sequence starting from 1, rather than 0. This can be an important consideration when working with certain types of data or mathematical models, particularly those that involve indexing or counting from a specific starting point.

Sum = 100(100+1)/2
= 100 x 101 / 2
= 5050

Therefore, the sum of the first 100 positive integers is indeed 5050.
```


In this example, TextGrad uses an **LLM-generated critique** to iteratively refine the answer. Over multiple iterations, the model can **self-improve** based on textual feedback.

---

### **Example 02: Smarter math calculations**

```python
import textgrad as tg
MODEL_NAME = "ollama/deepseek-r1"
engine = get_engine(f"experimental:{MODEL_NAME}", cache=False)

tg.set_backward_engine(engine=engine, override=True)
model = tg.BlackboxLLM(engine=engine)
question_string = ("If it takes 1 hour to dry 25 shirts under the sun, "
                   "how long will it take to dry 30 shirts under the sun? "
                   "Reason step by step")
question = tg.Variable(question_string, 
                       role_description="question to the LLM", 
                       requires_grad=False)
answer = model(question)
print(answer)
```
I used gemma and deepseek here. They both missed the trick.
It should not take any longer provided there is space for all the shirts to dry in parallel
```
<think>
First, I recognize that drying 25 shirts takes one hour. To find out how long it takes to dry one shirt, I divide the total time by the number of shirts: 1 hour divided by 25 shirts equals 4 minutes per shirt.

Next, I calculate the total time required to dry 30 shirts by multiplying the time per shirt by the number of shirts: 30 shirts multiplied by 4 minutes per shirt results in 120 minutes. Converting 120 minutes into hours gives me 2 hours.

Therefore, it will take 2 hours to dry 30 shirts under the sun.
</think>

**Solution:**

To determine how long it takes to dry 30 shirts, let's follow these steps:

1. **Determine the time per shirt:**
   
   - It takes **1 hour** to dry **25 shirts**.
   - To find the drying time for **1 shirt**, divide the total time by the number of shirts:
     
     \[
     \text{Time per shirt} = \frac{1\ \text{hour}}{25\ \text{shirts}} = 0.04\ \text{hours per shirt}
     \]

2. **Calculate the total drying time for 30 shirts:**
   
   - Multiply the time per shirt by the number of shirts:
     

3. **Final Answer:**

   It will take \(\boxed{2}\) hours to dry 30 shirts under the sun.
```


### **Example 03: Validate/enhance results**

Here I get the model to validate the results
```python
import textgrad as tg

# Define an initial question
question = tg.Variable("What is the capital of France?", 
                       role_description="question for llm",
                       requires_grad=False)

MODEL_NAME = "ollama/gemma"
engine = get_engine(f"experimental:{MODEL_NAME}", cache=False)
model = tg.BlackboxLLM(engine=engine)

answer = model(question)
print("Initial Answer:", answer.value)

# Define textual feedback as a loss function
evaluation_instruction = "Evaluate if the response correctly answers the question with context."
loss_fn = tg.TextLoss(evaluation_instruction)

# Compute the loss based on textual feedback
loss = loss_fn(answer)
print("Loss Feedback:", loss.value)

# Perform backpropagation and optimize the response
loss.backward()
optimizer = tg.TGD(parameters=[answer])
optimizer.step()

# Print the refined answer
print("Optimized Answer:", answer.value)

```

```
Loss Feedback: **The response is correct.**
The sentence clearly states that the capital of France is Paris, and also includes a French flag emoji to indicate the country.

Optimized Answer: The capital of France is Paris. Paris is the vibrant and historic capital of France, known for its stunning architecture, rich history, and iconic landmarks. Designated as the capital in the 18th century, Paris has played a pivotal role in shaping France's destiny.
```


### Example 04: Multimodal optimization
![bee](/img/bee.jpg)  

In this example I will use it to better classify an image.
then we will `check` result classification is good and `enhance` that result.

```python
from PIL import Image
import textgrad as tg
from textgrad.autograd import MultimodalLLMCall
from textgrad.loss import ImageQALoss

tg.set_backward_engine("gpt-4o")

image_path = "bee.jpg"
# Read the local image file in binary mode
with open(image_path, 'rb') as file:
    image_data = file.read()

# create the variable
image_variable = tg.Variable(image_data, role_description="image to answer a question about", requires_grad=False)

question_variable = tg.Variable("What do you see in this image?", role_description="question", requires_grad=False)
response = MultimodalLLMCall("gpt-4o")([image_variable, question_variable])
response
```

```
Variable(value=This image shows a close-up of a honeybee collecting pollen. 
The bee is perched on a cluster of flowers, and you can see pollen attached to its hind legs. 
The details of the bee's body, including its wings, eyes, and fuzzy thorax, are clearly visible., 
role=response from the language model, grads=set())
```
Lets get it to review its current answer.

```python
loss_fn = ImageQALoss(
    evaluation_instruction="""Does this seem like a complete and good answer for the image? 
                              Criticize. Do not provide a new answer.""",
    engine="gpt-4o"
)
loss = loss_fn(question=question_variable, image=image_variable, response=response)
loss

```


```
Variable(value=The answer is mostly complete and accurate. 
It correctly identifies the main elements of the image: a honeybee collecting pollen, 
the presence of flowers, and the details of the bee's body. 
However, it could be improved by mentioning the specific type of flowers or plant if identifiable, 
and by describing the setting or background to provide more context. Additionally, 
it could note the color and texture details of the bee and flowers for a more vivid description., 
role=evaluation of the response from the language model, grads=set())
```

Now lets get it to improve its answer.

```python
optimizer = tg.TGD(parameters=[response])
loss.backward()
optimizer.step()
print(response.value)
```

```
This image shows a close-up of a honeybee collecting pollen. 
The bee is perched on a cluster of flowers, possibly from a plant like goldenrod, with pollen visibly attached to its hind legs. 
The details of the bee's body, including its translucent wings, large compound eyes, and fuzzy thorax, are clearly visible. 
The bee's distinctive black and yellow stripes add to its vibrant appearance. 
The background appears to be a natural setting, with earthy tones that complement the scene. 
This moment captures the bee's vital role in pollination, as it diligently gathers pollen, contributing to the ecosystem. 
The gentle hum of its wings adds a sense of life and movement to the image.
```



### Example 05: Running with local models

textgrad examples are mainly OpenAI as I mentioned earlier this will cost you a lot of api calls.

I found the quality noticeably higher when using OpenAi across the board. 

It can be used with [LM Studio](https://lmstudio.ai/)
To use this you will need to download a model and have it running. It will be faster if you have a gpu.

Because it use LiteLLM it can also call [Ollama](https://ollama.com/) models

```python
from openai import OpenAI
from textgrad.engine.local_model_openai_api import ChatExternalClient
import textgrad as tg

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

engine = ChatExternalClient(client=client, model_string='mlabonne/NeuralBeagle14-7B-GGUF')

tg.set_backward_engine(engine, override=True)

initial_solution = """To solve the equation 3x^2 - 7x + 2 = 0, we use the quadratic formula:
x = (-b ± √(b^2 - 4ac)) / 2a
a = 3, b = -7, c = 2
x = (7 ± √((-7)^2 + 4(3)(2))) / 6
x = (7 ± √73) / 6
The solutions are:
x1 = (7 + √73)
x2 = (7 - √73)"""

solution = tg.Variable(initial_solution,
                       requires_grad=True,
                       role_description="solution to the math question")

loss_system_prompt = tg.Variable("""You will evaluate a solution to a math question. 
Do not attempt to solve it yourself, do not give a solution, only identify errors. Be super concise.""",
                                 requires_grad=False,
                                 role_description="system prompt")
                              
loss_fn = tg.TextLoss(loss_system_prompt)
optimizer = tg.TGD([solution])
loss = loss_fn(solution)
print(loss.value)
```

```
The given solution is correct and does not contain any errors. It correctly applies the quadratic formula to solve the equation 3x^2 - 7x + 2 = 0, and provides the two solutions x1 and x2 as calculated.
```

### **Example 06: Optimizing an LLM Response Using TextGrad**
Here’s a practical example using **TextGrad**:

```python
import textgrad as tg

# Define an initial question
question = tg.Variable("What is the capital of France?", 
                       role_description="question for llm",
                       requires_grad=False)

# Generate an initial response using GPT-4o
model = tg.BlackboxLLM("gpt-4o")
answer = model(question)
print("Initial Answer:", answer.value)

# Define textual feedback as a loss function
evaluation_instruction = "Evaluate if the response correctly answers the question with context."
loss_fn = tg.TextLoss(evaluation_instruction)

# Compute the loss based on textual feedback
loss = loss_fn(answer)
print("Loss Feedback:", loss.value)

# Perform backpropagation and optimize the response
loss.backward()
optimizer = tg.TGD(parameters=[answer])
optimizer.step()

# Print the refined answer
print("Optimized Answer:", answer.value)

```

#### **Expected Output**
1. **Initial Answer:**  
   - *"Paris is the capital of France."*
2. **LLM-Generated Feedback (Loss Function):**  
   - *"The response is correct but could include more context about France's geography and history."*   
3. **Optimized Answer (After Applying Feedback):**  
   - *"Paris is the capital and largest city of France, known for its rich history and cultural significance."*

```
Initial Answer: The capital of France is Paris.
Loss Feedback: The response correctly answers the question if the question was "What is the capital of France?" However, without the context of the question, it's unclear if this response is relevant.
Optimized Answer: In response to the question about France's capital, if you are asking about the capital of France, it is Paris. Paris is not only the capital but also a major European city known for its art, fashion, and culture. Could you please confirm if this is the information you were seeking?

```


---

## **Why This is Powerful**
- **Self-Improving AI**: Instead of manually tuning LLM prompts, **TextGrad automatically refines them**.
- **Works on Any Text-Based Task**: Can optimize **reasoning, coding solutions, drug discovery, and more**.
- **Mimics Backpropagation for LLMs**: Like gradient descent for neural networks, **TextGrad iteratively refines responses**.


---

## **The Power of TextGrad: Key Applications**
### **1. Adversarial Robustness in NLP**
Traditional NLP models struggle with **adversarial attacks**—textual inputs designed to trick AI models into incorrect predictions. **TextGrad enhances adversarial training by leveraging gradient-based textual modifications**, improving robustness against such attacks.

### **2. Prompt Engineering**
TextGrad can **automatically refine prompts** to enhance LLM performance. Instead of relying on manual prompt tuning, TextGrad iterates over possible modifications, optimizing clarity, structure, and effectiveness.

### **3. Code Optimization**
TextGrad has shown success in **improving AI-generated code solutions**. By using textual gradients to refine incorrect code snippets, TextGrad improves correctness and efficiency in solving programming challenges.

### **4. Drug Discovery and Molecular Optimization**
Beyond NLP, TextGrad has been applied to **molecular design**, optimizing chemical structures based on feedback from **LLM-powered evaluations**. This has significant implications for **AI-driven drug discovery**.

---

## Considerations when using TextGrad

* LLMs can be slow. This solution will involve multiple LLM calls.  
Increased latency due to multiple LLM calls.
* If you are paying for API calls this could be an expensive solution.


---

## **Beyond TextGrad: REVOLVE and Semantic Backpropagation**
While TextGrad introduced textual differentiation, recent advancements have built upon this foundation to **improve optimization stability and convergence**.

### **1. REVOLVE: Tracking Response Evolution**
One of TextGrad’s limitations is its reliance on **immediate feedback**—it optimizes based only on the most recent output, sometimes getting stuck in **suboptimal solutions**. **REVOLVE** addresses this by introducing a **long-term optimization strategy** that tracks how responses evolve over multiple iterations.

**Key Advantages of REVOLVE:**
- **Avoids Local Minima**: By analyzing response trends over time, it prevents premature convergence.
- **Adaptive Adjustments**: It fine-tunes responses progressively, leading to **higher quality outputs**.
- **Faster Convergence**: Reduces the number of iterations needed for optimization, saving computational resources.

> **Analogy**: If TextGrad is like using **first-order gradient descent**, then REVOLVE is akin to **second-order optimization**, incorporating curvature-like adjustments.

---

### **2. Semantic Backpropagation: Correcting Feedback Assignment**
Another limitation of standard TextGrad is **feedback misalignment**—it often fails to correctly assign feedback to the most relevant system components. **Semantic Backpropagation** solves this by improving **gradient propagation in agent-based AI systems**.

#### **Key Contributions:**
- **Neighbor-Aware Gradients**: Unlike TextGrad, which assigns feedback directly, **Semantic Backpropagation considers the dependencies between AI agents**, leading to **more accurate optimization**.
- **Graph-Based Optimization**: Treats AI systems as **computational graphs**, ensuring **feedback is distributed efficiently**.
- **Better Performance on QA Benchmarks**: It significantly outperforms existing methods on complex reasoning tasks.

---

## **The Future of Text-Based Optimization**
TextGrad and its successors represent a fundamental shift in **how AI systems are optimized**. Instead of treating LLMs as **static tools**, these methods **actively refine and improve AI-generated content** using **iterative textual feedback**.

---


## References


🔗 **[GitHub: TextGrad](https://github.com/zou-group/textgrad)**

🔗 **[TextGrad: Automatic "Differentiation" via Text]**(https://arxiv.org/abs/2406.07496v1)


🔗 **[Revolve: Optimizing AI Systems by Tracking Response Evolution in Textual Optimization]**(https://arxiv.org/abs/2412.03092)

---
