+++
date = '2025-02-09T22:23:08Z'
draft = true
title = 'Automating the Search for Artificial Life with Foundation Models'
+++

### Summary

Artificial Life (ALife) is a fascinating field that explores the nature of life through computational simulations. Traditionally, discovering interesting ALife simulations has been a manual and time-consuming process, relying heavily on trial and error. However, a recent paper titled *"Automating the Search for Artificial Life with Foundation Models"* introduces a groundbreaking approach called **Automated Search for Artificial Life (ASAL)**. This method leverages vision-language foundation models (FMs) to automate the discovery of lifelike simulations, opening up new possibilities for ALife research.

In this blog post, we’ll break down the key ideas from the paper and provide Python examples to help you understand how ASAL works and how you can apply similar techniques in your own projects.

## What is ASAL?

ASAL is a framework that uses vision-language foundation models (like CLIP) to search for interesting ALife simulations. The framework can:

1. **Find simulations that produce target phenomena** (e.g., a simulation that looks like a flock of birds).
2. **Discover simulations that generate open-ended novelty** (e.g., simulations that keep evolving in interesting ways over time).
3. **Illuminate a diverse set of simulations** (e.g., discovering a wide variety of lifelike behaviors).

The key innovation is that ASAL uses foundation models to evaluate the "interestingness" of simulations in a way that aligns with human perception. This allows researchers to automate the search for lifelike behaviors across a wide range of ALife substrates, such as Boids, Particle Life, and Lenia.

---

## Python Example: Searching for Target Phenomena

Let’s start with a simple example of how ASAL can search for simulations that produce a target phenomenon. We’ll use the **Boids** substrate, which simulates flocking behavior in bird-like objects.

### Step 1: Install Required Libraries

First, let’s install the necessary libraries. We’ll use `numpy` for simulation and `clip` for the foundation model.

```bash
pip install numpy torch torchvision clip
```

### Step 2: Define the Boids Simulation

Here’s a basic implementation of the Boids algorithm:

```python
import numpy as np

class Boids:
    def __init__(self, num_boids=128, width=500, height=500):
        self.num_boids = num_boids
        self.width = width
        self.height = height
        self.positions = np.random.rand(num_boids, 2) * np.array([width, height])
        self.velocities = np.random.rand(num_boids, 2) * 2 - 1

    def update(self):
        # Simple flocking rules: cohesion, alignment, and separation
        cohesion_strength = 0.01
        alignment_strength = 0.05
        separation_strength = 0.1

        # Cohesion: Move towards the center of mass
        center_of_mass = np.mean(self.positions, axis=0)
        cohesion = (center_of_mass - self.positions) * cohesion_strength

        # Alignment: Align with the average velocity of neighbors
        average_velocity = np.mean(self.velocities, axis=0)
        alignment = (average_velocity - self.velocities) * alignment_strength

        # Separation: Avoid crowding neighbors
        separation = np.zeros_like(self.positions)
        for i in range(self.num_boids):
            distances = np.linalg.norm(self.positions[i] - self.positions, axis=1)
            too_close = distances < 25
            too_close[i] = False  # Ignore self
            if np.any(too_close):
                separation[i] = -np.mean(self.positions[too_close] - self.positions[i], axis=0) * separation_strength

        # Update velocities and positions
        self.velocities += cohesion + alignment + separation
        self.positions += self.velocities

        # Wrap around the edges of the screen
        self.positions %= np.array([self.width, self.height])

    def render(self):
        # Render the boids as points on a 2D plane
        import matplotlib.pyplot as plt
        plt.scatter(self.positions[:, 0], self.positions[:, 1], s=10)
        plt.xlim(0, self.width)
        plt.ylim(0, self.height)
        plt.show()
```

### Step 3: Use CLIP to Evaluate the Simulation

Now, let’s use the CLIP model to evaluate how well the Boids simulation matches a target prompt, such as "a flock of birds."

```python
import torch
import clip
from PIL import Image
import matplotlib.pyplot as plt

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Define a function to evaluate the simulation
def evaluate_simulation(boids, target_prompt):
    # Render the simulation as an image
    plt.scatter(boids.positions[:, 0], boids.positions[:, 1], s=10)
    plt.xlim(0, boids.width)
    plt.ylim(0, boids.height)
    plt.savefig("boids.png")
    plt.close()

    # Preprocess the image and prompt
    image = preprocess(Image.open("boids.png")).unsqueeze(0).to(device)
    text = clip.tokenize([target_prompt]).to(device)

    # Compute the similarity score
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        similarity = (image_features @ text_features.T).item()

    return similarity

# Run the simulation and evaluate it
boids = Boids()
for _ in range(100):  # Run for 100 steps
    boids.update()

similarity = evaluate_simulation(boids, "a flock of birds")
print(f"Similarity to 'a flock of birds': {similarity}")
```

In this example, we use CLIP to evaluate how well the Boids simulation matches the target prompt "a flock of birds." The similarity score gives us a quantitative measure of how lifelike the simulation appears.

---

## Python Example: Searching for Open-Ended Novelty

Another key feature of ASAL is its ability to discover simulations that generate open-ended novelty. Let’s explore how this works using a simple **Cellular Automata (CA)** substrate.

### Step 1: Define a Cellular Automata Simulation

Here’s a basic implementation of a 2D cellular automata:

```python
class CellularAutomata:
    def __init__(self, size=64):
        self.size = size
        self.grid = np.random.choice([0, 1], size=(size, size))

    def update(self):
        new_grid = np.zeros_like(self.grid)
        for i in range(self.size):
            for j in range(self.size):
                # Count the number of alive neighbors
                neighbors = self.grid[max(i-1, 0):min(i+2, self.size), max(j-1, 0):min(j+2, self.size)]
                alive = np.sum(neighbors) - self.grid[i, j]
                # Apply Conway's Game of Life rules
                if self.grid[i, j] == 1 and (alive < 2 or alive > 3):
                    new_grid[i, j] = 0
                elif self.grid[i, j] == 0 and alive == 3:
                    new_grid[i, j] = 1
                else:
                    new_grid[i, j] = self.grid[i, j]
        self.grid = new_grid

    def render(self):
        import matplotlib.pyplot as plt
        plt.imshow(self.grid, cmap='binary')
        plt.show()
```

### Step 2: Measure Open-Endedness with CLIP

To measure open-endedness, we can use CLIP to evaluate how novel the simulation is over time. Here’s how:

```python
def measure_open_endedness(ca, steps=100):
    previous_states = []
    novelty_scores = []

    for step in range(steps):
        ca.update()
        ca.render()

        # Save the current state as an image
        plt.imshow(ca.grid, cmap='binary')
        plt.savefig(f"ca_step_{step}.png")
        plt.close()

        # Compute the CLIP embedding of the current state
        image = preprocess(Image.open(f"ca_step_{step}.png")).unsqueeze(0).to(device)
        with torch.no_grad():
            current_embedding = model.encode_image(image)

        # Compare with previous states to measure novelty
        novelty = 1.0
        if previous_states:
            similarities = [torch.dot(current_embedding, prev_embedding).item() for prev_embedding in previous_states]
            novelty = 1 - max(similarities)  # Novelty is the inverse of the maximum similarity

        novelty_scores.append(novelty)
        previous_states.append(current_embedding)

    return novelty_scores

# Run the CA and measure open-endedness
ca = CellularAutomata()
novelty_scores = measure_open_endedness(ca)
print(f"Novelty scores over time: {novelty_scores}")
```

In this example, we measure how novel the cellular automata simulation is over time by comparing each state to previous states using CLIP embeddings. High novelty scores indicate that the simulation is generating open-ended, interesting behaviors.

---

## Conclusion

The ASAL framework represents a significant leap forward in ALife research by automating the discovery of lifelike simulations using foundation models. By leveraging models like CLIP, researchers can now search for target phenomena, discover open-ended novelty, and illuminate diverse simulations with unprecedented ease.

In this blog post, we’ve explored how ASAL works and provided Python examples to help you get started with similar techniques. Whether you’re interested in simulating flocking behavior, cellular automata, or other ALife substrates, the combination of foundation models and automated search opens up exciting new possibilities for exploration.

To learn more about ASAL, check out the [project website](https://asal.sakana.ai/) and the [GitHub repository](https://github.com/SakanaAI/asal).

---

Feel free to modify the examples and expand on the concepts to suit your audience. Happy coding!