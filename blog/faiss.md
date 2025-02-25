+++
date = '2025-02-17T08:16:15Z'
draft = false
title = 'Faiss: A Fast, Efficient Similarity Search Library'
tag=['faiss', 'K-Means', 'scikit-learn',  'Vector Databases']
categories=['AI', 'faiss', 'scikit-learn', 'vector']
+++

## Summary

Searching through massive datasets efficiently is a challenge, whether in image retrieval, recommendation systems, or semantic search. [Faiss](https://github.com/facebookresearch/faiss) (`Facebook AI Similarity Search`) is a powerful open-source library developed by Meta to handle high-dimensional similarity search at scale.

It's particularly well-suited for tasks like:

* **Image search:** Finding visually similar images in a large database.
* **Recommendation systems:** Recommending items (products, movies, etc.) to users based on their preferences.
* **Semantic search:** Finding documents or text passages that are semantically similar to a given query.
* **Clustering:** Grouping similar vectors together.


In many of the upcoming projects in this blog I will be using it. It is a good local developer solution.

## Why use Faiss 

* **Speed**: Its fast, really fast. So for example a 1 billion vector dataset can be searched in milliseconds using 8GB RAM.
* **History**: It has been in active development for 8 years is developed by `Meta` and is used widely.
* **Scalability**: You can use it on your local machine and it will scale to very large solutions. 
* **Open Source**: This is the most important consideration when working in this field.

There are quite a few alternative solutions I suggest choosing one, knows its strengths and weaknesses and work with them.

## Basic Concepts

### Vectors and Embeddings
**Vectors**: In Faiss, data is represented as dense vectors. These vectors can be embeddings from machine learning models (e.g., word embeddings, image embeddings).

**Embeddings**: embeddings are dense vector representations of data (such as words, images, or other entities) that capture their semantic meaning or relationships in a lower-dimensional space. These vectors allow complex data to be transformed into numerical formats that can be processed by algorithms, while preserving important contextual information. Essentially they allow us to find similar items by looking at the data in different ways.

**Similarity Search**: Given a query vector, the goal is to find the most similar vectors in a dataset. This is typically measured using metrics like `euclidean distance` or `cosine similarity`.

**Clustering**: Faiss can also be used to cluster vectors into groups (e.g., K-Means clustering).


## Key Features of Faiss

* **Efficient indexing:** Faiss supports multiple indexing structures, allowing trade-offs between speed, accuracy, and memory usage.
    - **Flat Index** `(IndexFlatL2)`: Best for small datasets or exact nearest neighbor search.  Performs a linear scan through all vectors.
    - **IVF** `(Inverted File Index)`: Scales well for large datasets with partitioning.  Partitions the dataset into Voronoi cells and uses a coarse quantizer to assign vectors to cells
    - **HNSW** `(Hierarchical Graphs)`: Optimized for low-latency approximate search. A graph-based index that efficiently navigates through the dataset to find nearest neighbors
    - **PQ** `(Product Quantization)`: Best for memory-efficient indexing in billion-scale datasets. Partitions the feature space into subspaces and quantizes each subspace independently
* **GPU acceleration:** Many indexing algorithms are optimized for GPUs, significantly accelerating search performance.
* **Clustering algorithms:** Faiss includes implementations of various clustering algorithms, such as k-means and hierarchical clustering.
* **Python and C++ APIs:** Faiss provides user-friendly Python and C++ APIs, making it easy to integrate into your projects.


## Basic Usage

```python
import faiss

import numpy as np
import faiss

# Set random seed for reproducibility
np.random.seed(123)

# Define dataset parameters
dimension = 100    # Number of features per vector
num_vectors = 1000  # Number of vectors in the dataset

# Generate random dataset
data = np.random.random((num_vectors, dimension)).astype('float32')

# Create a flat index (brute-force search)
index = faiss.IndexFlatL2(d) 

# Add vectors to the index
index.add(data) 

# Perform a k-nearest neighbor search
k = 5  # Number of nearest neighbors
x = np.random.random((1, d)).astype('float32')
distances, indices = index.search(x, k) 

print(f"Distances: {distances}")
print(f"Indices: {indices}") 
```


**Example: IVF Index**

```python
# Create an IVF Index (Inverted File Index) with a Flat quantizer
quantizer = faiss.IndexFlatL2(dimension)  # Base index for clustering
num_cells = 100  # Number of Voronoi clusters (partitions)

index = faiss.IndexIVFFlat(quantizer, dimension, num_cells) 

# Train the index (clustering the dataset)
index.train(data)  # This step is necessary before adding vectors!

# Add vectors to the index
index.add(data)

# Set the number of Voronoi cells to search (trade-off between speed & accuracy)
index.nprobe = 10  

# Search for nearest neighbors
distances, indices = index.search(data[:5], k=5)  # Query first 5 vectors
```

**GPU Acceleration:**

Faiss supports GPU acceleration, which can significantly speed up similarity search and clustering.

```python
import time

# Perform a CPU-based search
start_cpu = time.time()
distances_cpu, indices_cpu = index.search(data[:10], k=5)
end_cpu = time.time()
print(f"CPU Search Time: {end_cpu - start_cpu:.6f} seconds")

# Move the index to GPU
gpu_resources = faiss.StandardGpuResources() 
gpu_index = faiss.index_cpu_to_gpu(gpu_resources, 0, index)

# Perform a GPU-based search
start_gpu = time.time()
distances_gpu, indices_gpu = gpu_index.search(data[:10], k=5)
end_gpu = time.time()
print(f"GPU Search Time: {end_gpu - start_gpu:.6f} seconds")
```


## Image Retrieval with Faiss

Building a simple image retrieval system using Faiss.

### Step 1: Extract Features
Use a pre-trained model (e.g., ResNet) to extract features from images.

```python
from torchvision import models, transforms
from PIL import Image
import torch

# Load a pre-trained ResNet model
model = models.resnet50(pretrained=True).eval()

# Define preprocessing steps
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Extract features from an image
def extract_features(image_path):
    img = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        features = model(input_tensor).numpy().astype('float32')
    return features
```

### Step 2: Build the Index
Store the extracted features in a Faiss index.

```python
# Extract features for all images
image_paths = [...]  # List of image paths
features = np.vstack([extract_features(path) for path in image_paths])

# Create a FAISS index
index = faiss.IndexFlatL2(features.shape[1])
index.add(features)
```

### Step 3: Perform Search
Search for similar images based on a query image.

```python
query_image_path = "query_image.jpg"
query_features = extract_features(query_image_path)

# Search for nearest neighbors
k = 5
distances, indices = index.search(query_features, k)

# Retrieve similar images
similar_images = [image_paths[i] for i in indices[0]]
print("Similar Images:", similar_images)
```



## Clustering with Faiss

Faiss also provides implementations of various clustering algorithms, such as:

* [**k-means:**]({{< relref "post/kmeans.md" >}}) Partitions the dataset into k clusters.
* **HDBSCAN:** A density-based clustering algorithm that can discover clusters of varying shapes and sizes.

**Example: k-means Clustering**

```python
kmeans = faiss.Kmeans(d, k, niter=20) 
kmeans.train(xb) 
D, I = kmeans.index.search(xb, 1) 
```

**Advanced Usage:**

* **Memory mapping:** Load large datasets into memory efficiently using memory mapping.
* **Multi-threading:** Utilize multiple CPU cores for faster indexing and search operations.
* **Custom quantization schemes:** Experiment with different quantization schemes for improved accuracy and efficiency.

## Faiss Example using Ollama to generate embeddings

```python
import ollama
import numpy as np
import faiss

# Example text data
documents = [
    "Python is a programming language that lets you work quickly.",
    "Machine learning automates model building through data analysis.",
    "Artificial Intelligence (AI) is intelligence demonstrated by machines.",
    "Natural language processing (NLP) is a field of AI that focuses on human language interaction.",
]

# Generate embeddings using Ollama
def get_embeddings(documents, embedding_model="mxbai-embed-large"):
    embeddings = []
    for doc in documents:
        response = ollama.embeddings(model=embedding_model, prompt=doc)
        embeddings.append(response["embedding"])
    # Convert embeddings to a NumPy array
    embeddings = np.array(embeddings).astype("float32")
    return embeddings

embeddings = get_embeddings(documents)

# Define the dimensionality of the embeddings
dimension = embeddings.shape[1]  # Number of features in each embedding

print(f"dimension={dimension}")
# Create a FAISS index
index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity search

# Add embeddings to the index
index.add(embeddings)

# Query embedding (generate an embedding for the query)
query = "What is Natural Language Processing?"

def get_embedding(text, embedding_model="mxbai-embed-large") -> np.ndarray:
    response = ollama.embeddings(model=embedding_model, prompt=text)
    embeddings = []
    embeddings.append(response["embedding"])
    embeddings = np.array(embeddings).astype('float32')
    return embeddings


query_embedding = get_embedding(query)

# Search for the top-k most similar embeddings
k = 2  # Number of nearest neighbors to retrieve
distances, indices = index.search(query_embedding, k)

# Print results
query = "What is Natural Language Processing?"
print(f"Indices of similar documents for search term:\n\t '{query}':\t", indices)
for i in indices[0]:
    print(f"Document {i}: {documents[i]}")

print("Distances to similar documents:", distances)

```

```
Indices of similar documents for search term:
	 'What is Natural Language Processing?':	 [[3 2]]
Document 3: Natural language processing (NLP) is a field of AI that focuses on human language interaction.
Document 2: Artificial Intelligence (AI) is intelligence demonstrated by machines.
Distances to similar documents: [[ 81.82834 277.91656]]

```


## Best Practices
* **Choose the Right Index Type** : Use Flat indexes for small datasets and IVF+PQ for larger ones.
* **Normalize Vectors** : Normalize vectors to unit length if using cosine similarity.
* **Optimize Parameters** : Experiment with parameters like nlist and m (number of subvectors in PQ) to balance speed and accuracy.
* **Leverage GPUs** : Use GPU acceleration for large-scale datasets.
* **Preprocess Data** : Ensure vectors are preprocessed (e.g., Principal Component  Analysis PCA) before indexing. This can improve both indexing time and accuracy.
* **Optimize Query Speed**: Set index.nprobe dynamically based on dataset size.
* **Reduce Memory Usage**: Use Product Quantization (PQ) for large-scale datasets.
* **Leverage Multi-threading**: Enable faiss.omp_set_num_threads(n_threads) for CPU parallelism.

## References

[Faiss wiki](https://github.com/facebookresearch/faiss/wiki)

[Faiss Getting Started](https://github.com/facebookresearch/faiss/wiki/Getting-started)


