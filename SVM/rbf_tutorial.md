# Radial Basis Function (RBF) kernel 

This kernel is particularly popular due to its effectiveness in handling non-linear data.

### Introduction to RBF Kernel

The Radial Basis Function (RBF) kernel, also known as the Gaussian kernel, is widely used because of its locality and finite response along the entire x-axis. It is defined as follows:

\[ K(x, x') = \exp\left(-\gamma \|x - x'\|^2\right) \]

where:
- \( x, x' \) are two feature vectors.
- \( \gamma \) (often denoted by \( \frac{1}{2\sigma^2} \)) is a parameter that sets the 'spread' of the kernel. A larger \( \gamma \) value leads to a narrower peak in the kernel (closer fitting to the training data).

### Advantages of RBF Kernel

- **Flexibility**: Capable of representing complex boundaries due to its non-linear nature.
- **Few Hyperparameters**: Primarily controlled by \( \gamma \) and the regularization parameter \( C \).

### Tutorial with Python Code

For this tutorial, we will use Python's `scikit-learn` library, which provides an implementation of SVM with an RBF kernel. Make sure you have `scikit-learn` installed, or you can install it using pip:

```bash
pip install scikit-learn
```

#### Step 1: Import Libraries

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
```

#### Step 2: Prepare the Data

We'll use the famous Iris dataset, focusing on two classes for simplicity.

```python
# Load data
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features for visualization
y = iris.target

# Filter out only classes 0 and 1 for binary classification
X = X[y != 2]
y = y[y != 2]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

#### Step 3: Train the SVM Model

```python
# Create an SVM classifier with RBF kernel
model = svm.SVC(kernel='rbf', gamma=0.7, C=1.0)  # You can experiment with different gamma values
model.fit(X_train, y_train)
```

#### Step 4: Evaluate the Model

```python
# Predict the labels on the dataset
y_pred = model.predict(X_test)

# Print performance details
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))
```

#### Step 5: Visualize Decision Boundaries

```python
# Create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Plot decision boundaries
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.title('SVM with RBF Kernel')
plt.show()
```

This tutorial provides a basic introduction to using the RBF kernel with SVM in Python. Experimenting with different values of \( \gamma \) and \( C \) can help you understand how they affect the model's performance and decision boundary.