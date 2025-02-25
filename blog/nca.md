+++
date = '2025-01-27T14:41:41Z'
draft = true
title = 'NCA Neural Cellular Automata'
+++

### **Neural Cellular Automata (NCA)**

### Summary
**Neural Cellular Automata (NCA)** derive their rules from data, blending the rigidity of cellular automata with the flexibility of machine learning.


### **What are Neural Cellular Automata?**

In traditional CA, each cell's state at the next time step is determined by fixed rules based on its current state and the states of its neighbors. NCAs modify this by:

1. **Neural Transition Rules**: Using machine learning models, such as neural networks, to infer rules from data.
2. **Adapting Dynamically**: Adjusting the learned rules as new data is introduced, enabling more realistic and versatile simulations.
3. **Applications**: Modeling phenomena like fluid dynamics, forest fires, disease spread, and traffic patterns where data-driven insights enhance accuracy.

---

### **Implementing NCA: A Practical Example**

Let’s implement an LCA to simulate the spread of a "contagion" (e.g., fire or disease) in a 2D grid, where the transition rules are learned using a neural network.

#### **1. Problem Setup**
We aim to predict a cell’s next state based on its current state and the states of its neighbors.

---

#### **2. Data Generation**
To train our model, we generate synthetic data using predefined rules to mimic the spread of the contagion.

```python
import numpy as np
import matplotlib.pyplot as plt

def generate_data(grid_size, steps):
    """Generate data using predefined CA rules."""
    data = []
    labels = []
    grid = np.random.choice([0, 1], size=(grid_size, grid_size), p=[0.7, 0.3])

    for _ in range(steps):
        for y in range(1, grid_size - 1):
            for x in range(1, grid_size - 1):
                neighborhood = grid[y-1:y+2, x-1:x+2].flatten()
                data.append(neighborhood)
                labels.append(1 if neighborhood.sum() > 4 else 0)

        # Update grid for next step
        new_grid = np.zeros_like(grid)
        for y in range(1, grid_size - 1):
            for x in range(1, grid_size - 1):
                neighborhood = grid[y-1:y+2, x-1:x+2]
                new_grid[y, x] = 1 if neighborhood.sum() > 4 else 0
        grid = new_grid

    return np.array(data), np.array(labels)

# Generate dataset
data, labels = generate_data(grid_size=10, steps=100)
```

---

#### **3. Training the Model**
We use a simple feedforward neural network to learn the transition rules.

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define the neural network class
class TransitionRuleNet(nn.Module):
    def __init__(self):
        super(TransitionRuleNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(9, 32),  # 3x3 neighborhood
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)

# Prepare the data for PyTorch
data_tensor = torch.tensor(data, dtype=torch.float32)
labels_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

# Initialize the model, loss, and optimizer
model = TransitionRuleNet()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(data_tensor)
    loss = criterion(outputs, labels_tensor)
    loss.backward()
    optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
```

---

#### **4. Simulation with Learned Rules**
Use the trained model to predict the next state of each cell in a new grid and save an animated GIF of the results.

```python
from PIL import Image

# Initialize a new grid
grid_size = 10
grid = np.random.choice([0, 1], size=(grid_size, grid_size), p=[0.7, 0.3])
steps = 20
frames = []

# Simulate with learned rules
for step in range(steps):
    fig, ax = plt.subplots()
    ax.imshow(grid, cmap='binary')
    ax.set_title(f"Step {step}")
    ax.axis('off')
    plt.savefig(f"frame_{step}.png")
    plt.close()

    # Add the frame to the animation
    frame = Image.open(f"frame_{step}.png")
    frames.append(frame)

    new_grid = np.zeros_like(grid)
    for y in range(1, grid_size - 1):
        for x in range(1, grid_size - 1):
            neighborhood = grid[y-1:y+2, x-1:x+2].flatten()
            with torch.no_grad():
                state = model(torch.tensor(neighborhood, dtype=torch.float32))
                new_grid[y, x] = 1 if state > 0.5 else 0

    grid = new_grid

# Save the animation as a GIF
gif_path = "lca_simulation.gif"
frames[0].save(
    gif_path,
    save_all=True,
    append_images=frames[1:],
    duration=300,
    loop=0
)
print(f"Simulation saved as {gif_path}")
```

---

### **Key Takeaways**

1. **Flexibility**: LCAs can adapt to various datasets, making them versatile for simulating real-world phenomena.
2. **Combining Simplicity and Power**: By learning rules, LCAs retain the simplicity of CA while leveraging the predictive power of machine learning.
3. **Applications**: LCAs are applicable in biology, physics, epidemiology, and urban planning.

This example demonstrates how LCAs bridge the gap between traditional CA and modern data-driven techniques, enabling adaptive simulations with minimal predefined rules.



### References

[Understanding Multiple Neighborhood Cellular Automata](https://slackermanz.com/understanding-multiple-neighborhood-cellular-automata/)

