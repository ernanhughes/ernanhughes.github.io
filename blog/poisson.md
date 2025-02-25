+++
date = '2025-01-28T15:01:00Z'
title = 'Fast Poisson Disk Sampling in Arbitrary Dimensions'
categories = ['Cellular Automata']
tag = ['ca'] 
+++

### Summary

In this post I explore Robert Bridson's paper:
[Fast Poisson Disk Sampling in Arbitrary Dimensions](https://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph07-poissondisk.pdf) and provide an example python implementation. 
Additionally, I introduce an alternative method using `Cellular Automata` to generate Poisson disk distributions.

Poisson disk sampling is a widely used technique in computer graphics, particularly for applications like rendering, texture generation, and particle simulation. Its appeal lies in producing sample distributions with "blue noise" characteristics—random yet evenly spaced, avoiding clustering. 


#### **The Challenge with Poisson Disk Sampling**

Poisson disk sampling is an effective method for generating evenly distributed points while preventing clustering. A naive approach, known as dart-throwing, involves randomly placing points and rejecting those that violate a minimum distance constraint. However, this method is highly inefficient, especially in higher dimensions. Bridson’s Algorithm improves efficiency by using a structured grid-based approach, reducing computational complexity while maintaining an even distribution.


---

#### **Bridson’s Algorithm: A Fresh Approach**
Bridson's algorithm introduces a modification to dart throwing that ensures Poisson disk samples are generated in \( O(N) \) time for \( N \) samples, regardless of dimensionality. Here's how it works:

1. **Background Grid Initialization**: 
   The algorithm begins by initializing an \( n \)-dimensional grid. Each grid cell has a size smaller than \( r / \sqrt{n} \), ensuring that each cell contains at most one sample. This accelerates spatial searches when checking if a new sample violates the minimum distance requirement.

2. **Initial Sample Placement**: 
   A random initial sample is placed in the domain and added to an "active list" for processing.

3. **Iterative Sampling**: 
   While the active list isn’t empty, a random sample from the list is selected. Up to \( k \) candidate points are generated in the spherical annulus between \( r \) and \( 2r \) around the chosen sample. Each candidate is checked for distance constraints using the background grid:
   - If the candidate is valid, it is added to the sample set and the active list.
   - If no valid candidate is found after \( k \) attempts, the sample is removed from the active list.

This process continues until the active list is empty, producing the desired distribution.

---

### Python example of Poisson disk sampling

- Initialize a grid to store sample positions.
- Start with a random seed point and add it to an active list.
- Iteratively generate new points within an annular region around active samples.
- Check if new points satisfy the minimum distance condition.
- If valid, add them to the active list; otherwise, remove them.


```python

class PoissonDiskSampling:
    def __init__(self, width, height, radius, k, visualizer):
        self.width = width
        self.height = height
        self.radius = radius
        self.k = k
        self.cell_size = radius / math.sqrt(2)
        self.grid_width = int(math.ceil(width / self.cell_size))
        self.grid_height = int(math.ceil(height / self.cell_size))
        self.grid = [[-1 for _ in range(self.grid_height)] for _ in range(self.grid_width)]
        self.samples = []
        self.active_list = []
        self.visualizer = visualizer

    def get_cell_coordinates(self, x, y):
        return int(x // self.cell_size), int(y // self.cell_size)

    def is_valid_point(self, x, y):
        cell_x, cell_y = self.get_cell_coordinates(x, y)
        for i in range(max(0, cell_x - 2), min(self.grid_width, cell_x + 3)):
            for j in range(max(0, cell_y - 2), min(self.grid_height, cell_y + 3)):
                if self.grid[i][j] != -1:
                    neighbor = self.samples[self.grid[i][j]]
                    if math.dist((x, y), neighbor) < self.radius:
                        return False
        return True

    def generate_samples(self):
        initial_sample = (random.uniform(0, self.width), random.uniform(0, self.height))
        self.samples.append(initial_sample)
        self.active_list.append(0)
        cell_x, cell_y = self.get_cell_coordinates(*initial_sample)
        self.grid[cell_x][cell_y] = 0

        while self.active_list:
            index = random.choice(self.active_list)
            base_sample = self.samples[index]
            found = False

            for _ in range(self.k):
                angle = random.uniform(0, 2 * math.pi)
                distance_from_base = random.uniform(self.radius, 2 * self.radius)
                candidate_x = base_sample[0] + math.cos(angle) * distance_from_base
                candidate_y = base_sample[1] + math.sin(angle) * distance_from_base

                if 0 <= candidate_x < self.width and 0 <= candidate_y < self.height and self.is_valid_point(candidate_x, candidate_y):
                    candidate_index = len(self.samples)
                    self.samples.append((candidate_x, candidate_y))
                    self.active_list.append(candidate_index)
                    cell_x, cell_y = self.get_cell_coordinates(candidate_x, candidate_y)
                    self.grid[cell_x][cell_y] = candidate_index
                    found = True
                    self.visualizer.draw_samples([self.samples[-1]])
                    break

            if not found:
                self.active_list.remove(index)

```

#### Visualize the process using pygame

Here we use pygame to visulize the generated samples

- Config class with defaulted values.
- Has a timeout of 60 seconds
- Will generate an animated gif
- Designed to run in a `Jupyter notebook`

```python

import pygame
import random
import math
import json
from PIL import Image, ImageDraw
import numpy as np
import time

# Load configuration from file if available, otherwise use default values
CONFIG_FILE = "config.json"
def load_config():
    default_config = {
        "WIDTH": 800,
        "HEIGHT": 800,
        "RADIUS": 20,
        "K": 30,
        "FPS": 10,
        "TIMEOUT": 60,
        "METHOD": "ca",  # Options: "bridson" or "ca"
        "OUTPUT_GIF": "poisson_disk_ca.gif"
    }
    try:
        with open(CONFIG_FILE, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return default_config

config = load_config()
WIDTH, HEIGHT = config["WIDTH"], config["HEIGHT"]
RADIUS, K, FPS = config["RADIUS"], config["K"], config["FPS"]
OUTPUT_GIF = config["OUTPUT_GIF"]
METHOD = config["METHOD"].lower()
TIMEOUT = config["TIMEOUT"]  # Maximum time to run the simulation (in seconds)

class PygameVisualizer:
    def __init__(self, width, height, timeout=TIMEOUT):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Poisson Disk Sampling")
        self.screen.fill((255, 255, 255))
        self.clock = pygame.time.Clock()
        self.timeout = timeout
        self.frames = []

    def save_frame(self):
        pygame_surface = pygame.display.get_surface()
        raw_str = pygame.image.tostring(pygame_surface, "RGB")
        img = Image.frombytes("RGB", (WIDTH, HEIGHT), raw_str)
        self.frames.append(img)

    def draw_samples(self, samples):
        for x, y in samples:
            pygame.draw.circle(self.screen, (0, 0, 0), (int(x), int(y)), 3)
        self.save_frame()
        pygame.display.flip()

    def save_gif(self, output_gif):
        self.frames[0].save(output_gif, save_all=True, append_images=self.frames[1:], optimize=False, duration=1000 // FPS, loop=0)
        print(f"Animated GIF saved as {output_gif}")

    def run(self):
        running = True
        start_time = time.time()  # Record start time
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            elapsed_time = time.time() - start_time
            if elapsed_time >= self.timeout:
                print("Timeout reached. Closing visualization.")
                running = False
            self.clock.tick(FPS) # Ensure frame rate is respected even with timeout
        pygame.quit()


OUTPUT_GIF = 'poisson_disk.gif'
visualizer = PygameVisualizer(WIDTH, HEIGHT)
poisson_ca = PoissonDiskSampling(WIDTH, HEIGHT, RADIUS // 2, RADIUS, 100, visualizer)
poisson_ca.generate_samples()
# Save the GIF
visualizer.save_gif(OUTPUT_GIF)
# Run the visualizer
visualizer.run()

```

#### Generated result


![Poisson Disk Sampling](/img/poisson_disk_bridson.gif)


#### **Key Advantages**
- **Efficiency**: With linear time complexity, Bridson’s algorithm is significantly faster than naive approaches and scales well to higher dimensions.
- **Simplicity**: The method avoids complex geometric computations, relying instead on efficient rejection sampling within a manageable search space.
- **Generality**: The algorithm seamlessly adapts to any dimensional space, making it versatile for various applications like 3D rendering with motion blur or depth-of-field effects.

---

#### **Performance and Results**
Bridson’s paper demonstrates the algorithm's efficiency and effectiveness through examples in two and three dimensions. The resulting sample patterns exhibit the desired blue noise properties, verified through periodograms. These results underline the algorithm's practical utility in producing high-quality, evenly spaced samples for real-world applications.

---

#### **Applications and Implications**
The ability to efficiently generate Poisson disk samples in arbitrary dimensions opens up possibilities across numerous fields:
- **Graphics and Animation**: Improved rendering techniques for scenes with motion blur, soft shadows, or depth of field.
- **Particle Systems**: Realistic distribution of particles in simulations or procedural generation.
- **Data Sampling**: Applications in high-dimensional data spaces, such as machine learning or computational biology.

---


### Cellular Automata version

`Cellular Automata` (CA) offers a different approach, where points evolve over time based on simple activation rules. Instead of directly computing candidate positions, each cell in a grid decides whether it should activate based on its neighbors. This creates a more organic, step-by-step generation process, making it useful for procedural generation.


```python

class Cell:
    def __init__(self, x, y, cell_size, radius):
        self.x = x
        self.y = y
        self.cell_size = cell_size
        self.radius = radius
        self.active = False

    def should_activate(self, grid):
        radius_cells = int(self.radius / self.cell_size)
        for dy in range(-radius_cells, radius_cells + 1):
            for dx in range(-radius_cells, radius_cells + 1):
                nx, ny = self.x + dx, self.y + dy
                if 0 <= nx < len(grid[0]) and 0 <= ny < len(grid):
                    if grid[ny][nx].active:
                        dist = math.sqrt((nx - self.x) ** 2 + (ny - self.y) ** 2) * self.cell_size
                        if dist < self.radius:
                            return False
        return random.random() < 0.2

class CellularAutomata:
    def __init__(self, grid_width, grid_height, cell_size, radius):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.cell_size = cell_size
        self.radius = radius
        self.grid = [[Cell(x, y, cell_size, radius) for x in range(grid_width)] for y in range(grid_height)]

    def update(self):
        new_samples = []
        for row in self.grid:
            for cell in row:
                if not cell.active and cell.should_activate(self.grid):
                    cell.active = True
                    new_samples.append((cell.x * self.cell_size, cell.y * self.cell_size))
        return new_samples

class PoissonDiskSamplingCA:
    def __init__(self, width, height, cell_size, radius, steps, visualizer):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.radius = radius
        self.steps = steps
        self.visualizer = visualizer
        self.automata = CellularAutomata(width // cell_size, height // cell_size, cell_size, radius)
        self.samples = []

    def generate_samples(self):
        for _ in range(10):
            x = random.randint(0, self.automata.grid_width - 1)
            y = random.randint(0, self.automata.grid_height - 1)
            self.automata.grid[y][x].active = True
            self.samples.append((x * self.cell_size, y * self.cell_size))

        for step in range(self.steps):
            new_samples = self.automata.update()
            if new_samples:
                self.visualizer.draw_samples(new_samples)
                self.samples.extend(new_samples)
        return self.samples

```

1. **Cell Representation**:  
   - Each **Cell** object knows its **position (x, y), size, radius, and activation state**.  
   - It decides whether to activate based on nearby active cells and a **random probability (20%)**.

2. **Grid Initialization**:  
   - The **CellularAutomata** class creates a **grid** of `Cell` objects, with dimensions based on `grid_width` and `grid_height`.

3. **Activation Rules**:  
   - A **cell activates** if no nearby cells (within `radius`) are already active.  
   - Activation follows a **probabilistic rule** to ensure organic distribution.

4. **Grid Updates**:  
   - The `update()` method **iterates through all cells**, checking if they should activate.  
   - Newly activated cells are **added to the sample list**.

5. **Poisson Disk Sampling Process**:  
   - Starts with **10 randomly activated cells** as seeds.  
   - Iterates for `steps`, activating new valid cells at each step.

6. **Visualization & Sample Storage**:  
   - The **visualizer draws new samples** each step.  
   - All generated points are stored in `self.samples` for further use.

#### CA Result

![Poisson Disk Sampling](/img/poisson_disk_ca.gif)


### Supporting classes

#### Distribution validator

Initially I was getting really bad results using the CA so I wrote a method to validate the generated results

```python
def validate_distribution(samples, radius):
    """Validates the distribution of samples and checks radius constraint."""
    if not samples:
        return True  # No samples, trivially valid

    min_distances = []
    for i in range(len(samples)):
        min_dist = float('inf')
        for j in range(len(samples)):
            if i != j:
                dist = math.dist(samples[i], samples[j])
                min_dist = min(min_dist, dist)
        min_distances.append(min_dist)

    min_overall_distance = min(min_distances) if min_distances else 0
    print(f"Minimum distance between any two points: {min_overall_distance}")

    # Check if all distances are greater than or equal to the radius
    radius_valid = all(dist >= radius - 1e-6 for dist in min_distances) # Tolerance for floating point errors
    print(f"Radius constraint valid: {radius_valid}")
    
    # Analyze the distribution (example: check for clusters)
    # (More sophisticated analysis might be needed depending on your needs)
    distances_to_neighbors = []
    for i in range(len(samples)):
        for j in range(len(samples)):
            if i != j:
                distances_to_neighbors.append(math.dist(samples[i], samples[j]))
    
    avg_neighbor_distance = np.mean(distances_to_neighbors) if distances_to_neighbors else 0
    print(f"Average distance to nearest neighbor: {avg_neighbor_distance}")

    return radius_valid, min_overall_distance, avg_neighbor_distance  # Return values for further analysis


## use like this
if METHOD == "bridson":
    poisson = PoissonDiskSampling(WIDTH, HEIGHT, RADIUS, K, visualizer)
    poisson.generate_samples()
    radius_valid, min_dist, avg_neighbor_dist = validate_distribution(poisson.samples, RADIUS)
elif METHOD == "ca":
    poisson_ca = PoissonDiskSamplingCA(WIDTH, HEIGHT, RADIUS // 2, RADIUS, 1000, visualizer)
    poisson_ca.generate_samples()
    radius_valid, min_dist, avg_neighbor_dist = validate_distribution(poisson_ca.samples, RADIUS)
else:
    print("Invalid METHOD specified in config. Use 'bridson' or 'ca'.")

print(f"Poisson disk sampling (method: {METHOD}) validation:")
print(f"Radius constraint satisfied: {radius_valid}")
print(f"Minimum distance: {min_dist}")
print(f"Average nearest neighbor distance: {avg_neighbor_dist}")

```

### **Code Examples**

Check out the [programmer.ie notebooks](https://github.com/ernanhughes/programmer.ie.notebooks) for the code used in this post and additional examples.
