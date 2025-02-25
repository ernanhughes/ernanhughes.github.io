+++
date = '2025-02-06T08:44:07Z'
draft = false
title = 'Writing Neural Networks with PyTorch'
categories = ['Deep Learning', 'PyTorch', 'Neural Networks', 'AI']
tags = ['pytorch', 'neural networks', 'deep learning', 'AI', 'machine learning']
+++

### Summary

This post provides a practical guide to building common neural network architectures using PyTorch. We'll explore `feedforward` networks, `convolutional` neural networks (CNNs), `recurrent` neural networks (RNNs), `LSTM`s, `transformers`, `autoencoders`, and `GAN`s, along with code examples and explanations.


---

### **1. Understanding PyTorch's Neural Network Module**

PyTorch provides the `torch.nn` module to build neural networks. 
It provides classes for defining layers, [activation functions]({{< relref "post/activation.md" >}}), and loss functions, making it easy to create and manage complex network architectures in a structured way.

#### **Key Components:**
* `torch.nn.Module`: The base class for all neural network models.
* `torch.nn.Linear`: A fully connected (dense) layer.
* `torch.nn.ReLU`, `torch.nn.Sigmoid`, `torch.nn.Tanh`, `torch.nn.Softmax`: Common [activation functions]({{< relref "post/activation.md" >}}).
* `torch.optim`: Optimizers for training.
* `torch.nn.functional` (often imported as `F`): Contains activation functions, loss functions, and other utility functions.

---

### **2. Creating a Simple Feedforward Neural Network**

Let's build a basic neural network with PyTorch using `torch.nn.Module`.

#### **Step 1: Import Required Libraries**
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
```

#### **Step 2: Define the Neural Network**
```python
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # First hidden layer
        self.relu = nn.ReLU()  # Activation function
        self.fc2 = nn.Linear(hidden_size, output_size)  # Output layer
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
```

#### **Step 3: Create a Model Instance**
```python
input_size = 10
hidden_size = 5
output_size = 2

model = SimpleNN(input_size, hidden_size, output_size)
print(model) # Print the model architecture
```

```
SimpleNN(
  (fc1): Linear(in_features=10, out_features=5, bias=True)
  (relu): ReLU()
  (fc2): Linear(in_features=5, out_features=2, bias=True)
)

```

---

### **3. Training the Neural Network**

To train the neural network, we need:
- A **loss function** (e.g., Mean Squared Error for regression, Cross Entropy for classification).
- An **optimizer** (e.g., Stochastic Gradient Descent, Adam).
- A **training loop** to update model weights.

#### **Step 1: Define Loss Function and Optimizer**
```python
criterion = nn.CrossEntropyLoss()  # For classification
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer
```

#### **Step 2: Generate Some Dummy Data**
```python
# Generate random input and target data
x_train = torch.randn(100, input_size)  # 100 samples, 10 features each
y_train = torch.randint(0, output_size, (100,))  # 100 labels (0 or 1)
```

#### **Step 3: Training Loop**
```python
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()  # Clear previous gradients
    outputs = model(x_train)  # Forward pass
    loss = criterion(outputs, y_train)  # Compute loss
    loss.backward()  # Backpropagation
    optimizer.step()  # Update weights
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
```

```
Epoch [10/100], Loss: 0.7435
Epoch [20/100], Loss: 0.7355
Epoch [30/100], Loss: 0.7286
Epoch [40/100], Loss: 0.7226
Epoch [50/100], Loss: 0.7174
Epoch [60/100], Loss: 0.7127
Epoch [70/100], Loss: 0.7084
Epoch [80/100], Loss: 0.7041
Epoch [90/100], Loss: 0.6999
Epoch [100/100], Loss: 0.6956

```

---

### **4. Evaluating the Model**

After training, we need to evaluate the model's performance.

```python
with torch.no_grad():  # Disable gradient computation for evaluation
    test_inputs = torch.randn(10, input_size)
    test_outputs = model(test_inputs)
    predicted_labels = torch.argmax(test_outputs, dim=1)
    print("Predicted Labels:", predicted_labels)
```

```
Predicted Labels: tensor([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
```

---

### **5. Using Activation Functions**

PyTorch provides multiple activation functions. Here’s an example of how to use them:

```python
class ActivationNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ActivationNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
```

---

### **6. Saving and Loading Models**

Saving models is essential for reusing trained networks.

#### **Saving the Model**
```python
torch.save(model.state_dict(), 'model.pth')
```

#### **Loading the Model**
```python
model.load_state_dict(torch.load('model.pth'))
model.eval()  # Set model to evaluation mode
```

```
SimpleNN(
  (fc1): Linear(in_features=10, out_features=5, bias=True)
  (relu): ReLU()
  (fc2): Linear(in_features=5, out_features=2, bias=True)
)
```


---


### Different types of Neural Networks

Neural networks can be categorized based on their **architecture, functionality, and use cases**. Here are the main types:

---

#### **1. Feedforward Neural Networks (FNNs)**
- The simplest type of neural network.
- Information moves in one direction: from input to output (no loops).
- Used for tasks like **classification** and **regression**.

**Example in PyTorch:**
```python
class FeedforwardNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FeedforwardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
model = FeedforwardNN(input_size, hidden_size, output_size)
print(model)
```

```
FeedforwardNN(
  (fc1): Linear(in_features=10, out_features=5, bias=True)
  (relu): ReLU()
  (fc2): Linear(in_features=5, out_features=2, bias=True)
)

```

---

#### **2. Convolutional Neural Networks (CNNs)**
- Designed for image processing and computer vision tasks.
- Uses convolutional layers to **extract spatial features** from images.
- Includes pooling layers to reduce dimensionality.

**Example in PyTorch:**
```python
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 14 * 14, 10)  # Fully connected layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.fc1(x)
        return x

model = CNN()
print(model)
```

```
CNN(
  (conv1): Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (relu): ReLU()
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fc1): Linear(in_features=3136, out_features=10, bias=True)
)
```

---

#### **3. Recurrent Neural Networks (RNNs)**
- Designed for **sequential data** (e.g., time series, natural language processing).
- Uses **hidden states** to retain information from previous steps.

**Example in PyTorch:**
```python
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # Get the last time step output
        return out

model = RNN(input_size=10, hidden_size=5, output_size=2)
print(model)

```
```
RNN(
  (rnn): RNN(10, 5, batch_first=True)
  (fc): Linear(in_features=5, out_features=2, bias=True)
)
```
---

#### **4. Long Short-Term Memory Networks (LSTMs)**
- A special type of RNN designed to handle **long-term dependencies**.
- Uses **gates (input, forget, and output)** to regulate information flow.

**Example in PyTorch:**
```python
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

model = LSTM(input_size=10, hidden_size=5, output_size=2)
print(model)
```

```
LSTM(
  (lstm): LSTM(10, 5, batch_first=True)
  (fc): Linear(in_features=5, out_features=2, bias=True)
)

```

---

#### **5. Transformers**
- Used in **Natural Language Processing (NLP)** (e.g., BERT, GPT).
- Replaces RNNs/LSTMs with **self-attention mechanisms**.
- Scales well to large datasets.

**Example using PyTorch's `transformers` library:**
```python
from transformers import BertModel

bert_model = BertModel.from_pretrained("bert-base-uncased")
print(bert_model)
```

```
BertModel(
  (embeddings): BertEmbeddings(
    (word_embeddings): Embedding(30522, 768, padding_idx=0)
    (position_embeddings): Embedding(512, 768)
    (token_type_embeddings): Embedding(2, 768)
    (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
    (dropout): Dropout(p=0.1, inplace=False)
  )
  (encoder): BertEncoder(
    (layer): ModuleList(
      (0-11): 12 x BertLayer(
        (attention): BertAttention(
          (self): BertSdpaSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): BertSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): BertIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
          (intermediate_act_fn): GELUActivation()
        )
        (output): BertOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
    )
  )
  (pooler): BertPooler(
    (dense): Linear(in_features=768, out_features=768, bias=True)
    (activation): Tanh()
  )
)

```
---

#### **6. Autoencoders**
- Used for **dimensionality reduction, anomaly detection, and generative models**.
- Consists of an **encoder** (compressing input) and a **decoder** (reconstructing input).

**Example in PyTorch:**
```python
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Linear(784, 128)
        self.decoder = nn.Linear(128, 784)

    def forward(self, x):
        x = torch.relu(self.encoder(x))
        x = torch.sigmoid(self.decoder(x))
        return x

model = Autoencoder()
print(model)

```
```
Autoencoder(
  (encoder): Linear(in_features=784, out_features=128, bias=True)
  (decoder): Linear(in_features=128, out_features=784, bias=True)
)

```
---

#### **7. Generative Adversarial Networks (GANs)**
- Used for **image generation** (e.g., DeepFake, AI art).
- Composed of:
  - **Generator** (creates fake samples).
  - **Discriminator** (determines real vs. fake samples).

**Example in PyTorch:**
```python
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc = nn.Linear(100, 784)

    def forward(self, x):
        return torch.tanh(self.fc(x))

model = Generator()
print(model)
```

```
Generator(
  (fc): Linear(in_features=100, out_features=784, bias=True)
)

```

---

### **Summary of Neural Network Types**
| Type | Best For |
|------|---------|
| **Feedforward NN** | General tasks, structured data |
| **CNN** | Image recognition, computer vision |
| **RNN** | Sequential data, speech processing |
| **LSTM** | Long-term memory retention (e.g., chatbots) |
| **Transformers** | NLP, large-scale text tasks |
| **Autoencoders** | Dimensionality reduction, anomaly detection |
| **GANs** | Image generation, unsupervised learning |

