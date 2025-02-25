+++
date = '2025-02-19T14:48:58Z'
draft = false
title = 'RAFT: Reward rAnked FineTuning - A New Approach to Generative Model Alignment'
categories = ['RAFT', 'CLIP', 'RLHF', 'fine-tuning']
tag = ['RAFT', 'CLIP', 'RLHF', 'fine-tuning'] 
+++

## Summary

This post is an explanation of this paper:[RAFT: Reward rAnked FineTuning for Generative Foundation Model Alignment](https://arxiv.org/abs/2304.06767).

Generative foundation models, such as **Large Language Models (LLMs)** and **diffusion models**, have revolutionized AI by achieving human-like content generation. 
However, they often suffer from
1. Biases – Models can learn and reinforce societal biases present in the training data (e.g., gender, racial, or cultural stereotypes).
2. Ethical Concerns – AI-generated content can be misused for misinformation, deepfakes, or spreading harmful narratives.
3. Alignment Issues – The model’s behavior may not match human intent, leading to unintended or harmful outputs despite good intentions.

Traditionally, **Reinforcement Learning from Human Feedback (RLHF)** has been used to align these models, but RLHF comes with stability and efficiency challenges.
To address these limitations, **RAFT (Reward rAnked FineTuning)** was introduced as a more stable and scalable alternative. RAFT fine-tunes models using a **ranking-based approach** to filter high-reward samples, allowing generative models to improve without complex reinforcement learning setups.

---

## **Reinforcement Learning from Human Feedback (RLHF)**  

**RHLF** is a technique used to fine-tune AI models by incorporating **human preferences** to improve their responses.  
It helps align AI-generated content with human expectations, making it safer, more reliable, and useful.

### **How RLHF Works**  

1. **Pretraining the Model** 
   * The AI model (e.g., GPT) is first trained using massive amounts of text data (unsupervised learning).    
2. **Collecting Human Feedback**
   * Humans provide feedback on multiple AI-generated responses for a given prompt.  
   * They rank or label which responses are better in terms of quality, correctness, and safety.  
3. **Training a Reward Model**  
   * A **reward model** is trained using the ranked feedback to predict which responses are preferred.  
   * The reward model helps the AI understand what humans consider “good” responses.  
4. **Fine-Tuning with Reinforcement Learning (PPO Algorithm)**
   * The AI model is further fine-tuned using **Proximal Policy Optimization (PPO)**, a reinforcement learning algorithm.  
   * The model generates responses, the reward model scores them, and the AI updates itself to maximize high-reward outputs.  
5. **Iterate and Improve**  
   - This process is repeated iteratively to improve model alignment with human expectations.  

### **Why Use RLHF?**  

✅ **Better Alignment** – Helps AI produce more **helpful and ethical** responses.  
✅ **Reduces Harmful Outputs** – Minimizes **biases, toxic language, and unsafe content**.  
✅ **More Human-Like Responses** – AI becomes **more natural and engaging** in conversations.  

### **Limitations of RLHF**  

❌ **Expensive & Time-Consuming** – Requires a **large amount of human feedback**.  
❌ **Bias Transfer** – If human feedback is biased, the model **inherits** those biases.  
❌ **Instability** – RLHF can sometimes cause models to **over-optimize**, leading to strange or unnatural behavior.  

---

## How RAFT Works
RAFT operates in an **iterative three-step process**:
1. **Data Collection**: Generate a batch of samples using the generative model.
2. **Data Ranking**: Score the samples using a reward model and select high-quality responses.
3. **Fine-Tuning**: Retrain the generative model on the top-ranked samples.

This cycle continues until the model achieves convergence, allowing it to improve iteratively.

### Key Advantages of RAFT over RLHF
- **No complex reinforcement learning (RL) algorithms**: Unlike PPO-based RLHF, RAFT avoids stability issues associated with policy gradient methods.
- **Lower memory usage**: RAFT does not require multiple models running in parallel (e.g., critic models, reference models, etc.), making it computationally efficient.
- **Better interpretability**: The model learns from high-reward samples directly, making the learning process easier to monitor and debug.

---
## Python Implementation: RAFT from Scratch

### **Step 1: Training a Reward Model**
The reward model evaluates generated responses, assigning scores that help filter high-quality outputs.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load a pre-trained model for reward estimation
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
reward_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=1)

# Sample training data (prompt-response pairs with human-labeled scores)
data = [
    ("What is AI?", "AI stands for Artificial Intelligence.", 0.9),
    ("What is deep learning?", "Deep learning is a subset of machine learning.", 0.8),
    ("Tell me a joke.", "Why did the chicken cross the road?", 0.6)
]

# Convert data into tensors
inputs = tokenizer([d[1] for d in data], padding=True, truncation=True, return_tensors="pt")
labels = torch.tensor([d[2] for d in data]).unsqueeze(1)

# Define optimizer and loss function
optimizer = optim.AdamW(reward_model.parameters(), lr=5e-5)
criterion = nn.MSELoss()

# Training loop
reward_model.train()
for epoch in range(3):
    optimizer.zero_grad()
    outputs = reward_model(**inputs).logits
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
```

```
Epoch 1, Loss: 0.5691652297973633
Epoch 2, Loss: 0.34353339672088623
Epoch 3, Loss: 0.11656033992767334
```

---
### **Step 2: Implementing RAFT for Fine-Tuning**
Next, we implement the RAFT framework, which iteratively selects the best-generated samples and fine-tunes a language model.

```python
from transformers import AutoModelForCausalLM
import random

# Load a small generative model (e.g., GPT-2)
generative_model = AutoModelForCausalLM.from_pretrained("gpt2")

def generate_samples(prompt, num_samples=5):
    """Generate multiple responses for a given prompt."""
    return [f"Generated response {i} for {prompt}" for i in range(num_samples)]

def rank_samples(samples):
    """Rank samples using the reward model."""
    inputs = tokenizer(samples, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        rewards = reward_model(**inputs).logits.squeeze()
    return [samples[i] for i in torch.argsort(rewards, descending=True)]

def fine_tune_model(model, dataset):
    """Fine-tune the generative model on high-reward samples."""
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    
    for epoch in range(3):
        for sample in dataset:
            inputs = tokenizer(sample, return_tensors="pt")
            labels = inputs["input_ids"].clone()
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} Fine-tuning loss: {loss.item()}")

# RAFT loop
for iteration in range(5):
    print(f"Iteration {iteration+1}...")
    prompt = "What is artificial intelligence?"
    samples = generate_samples(prompt)
    ranked_samples = rank_samples(samples)[:3]  # Select top-3 responses
    fine_tune_model(generative_model, ranked_samples)
```

```
Iteration 1...
Epoch 1 Fine-tuning loss: 11.924062728881836
Epoch 2 Fine-tuning loss: 11.158195495605469
Epoch 3 Fine-tuning loss: 10.423539161682129
Iteration 2...
Epoch 1 Fine-tuning loss: 8.566413879394531
Epoch 2 Fine-tuning loss: 8.518267631530762
Epoch 3 Fine-tuning loss: 7.131834506988525
Iteration 3...
Epoch 1 Fine-tuning loss: 6.632737159729004
Epoch 2 Fine-tuning loss: 5.000913143157959
Epoch 3 Fine-tuning loss: 4.756820201873779
Iteration 4...
Epoch 1 Fine-tuning loss: 4.7517194747924805
Epoch 2 Fine-tuning loss: 4.390162944793701
Epoch 3 Fine-tuning loss: 3.733025312423706
Iteration 5...
Epoch 1 Fine-tuning loss: 2.687434673309326
Epoch 2 Fine-tuning loss: 1.3824565410614014
Epoch 3 Fine-tuning loss: 1.3723318576812744
```



---
## Applying RAFT to Diffusion Models
RAFT can also be applied to **image generation models** such as **Stable Diffusion** by using **CLIP scores** as rewards.

```python
from transformers import CLIPProcessor, CLIPModel
import requests
from PIL import Image

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def compute_reward(image, text):
    """Compute the similarity score between an image and text."""
    inputs = clip_processor(text=[text], images=image, return_tensors="pt")
    scores = clip_model(**inputs).logits_per_image
    return scores.item()

# Example: Evaluating a generated image
image_url = "https://example.com/sample.jpg"
image = Image.open(requests.get(image_url, stream=True).raw)
reward_score = compute_reward(image, "A high-quality anime-style portrait")
print("Reward Score:", reward_score)
```
```
Reward Score: 27.507169723510742
```

![Anime Boy](/img/anime_boy.png)


### CLIP (Contrastive Language–Image Pre-training) 

Is a neural network developed by OpenAI that learns to understand the relationship between images and text. A CLIP score, therefore, represents how well an image aligns with a given textual description. 

* **Image and Text Embeddings:**
    * CLIP encodes images and text into high-dimensional vectors (embeddings).
    * These embeddings represent the semantic meaning of the image and text.
* **Similarity Measurement:**
    * The CLIP score is essentially a measure of the similarity between the image and text embeddings.
    * Higher scores indicate a stronger alignment between the image and the text.
* **How it Works:**
    * CLIP is trained on a massive dataset of image-text pairs.
    * During training, it learns to maximize the similarity of embeddings for matching image-text pairs and minimize the similarity for non-matching pairs.
* **Applications:**
    * CLIP scores are widely used in tasks like image retrieval, zero-shot image classification, and image generation evaluation.
    * In the context of the RAFT blog post, it is used to measure how well a generated image from a diffusion model matches a text prompt.
* **Intuition:**
    * Think of it as a way for a computer to "understand" if an image depicts what a text description says.
    * If the image and text are closely related, the CLIP score will be high; if they are unrelated, the score will be low.


---


**Key Points:**
- RAFT is **easier to implement** and requires **fewer resources** than PPO-based RLHF.
- It applies to both **language models and diffusion models**.
- By leveraging **reward ranking**, it provides better interpretability and robustness.


### **Code Examples**

Check out [raft](https://github.com/ernanhughes/programmer.ie.notebooks/blob/main/notebooks/raft.ipynb) for the code used in this post.
