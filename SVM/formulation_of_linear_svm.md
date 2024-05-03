Certainly! The mathematical formulation of the linear Support Vector Machine (SVM) algorithm is integral to understanding how it operates for classification tasks. Here’s a step-by-step breakdown of the formulation:

### Objective Function
The primary objective of a linear SVM is to find a hyperplane that best separates the data into two classes. In a two-dimensional space, this hyperplane is a line defined as:
\[ w \cdot x + b = 0 \]
Where:
- \( w \) is the weight vector (normal to the hyperplane).
- \( x \) represents the feature vectors.
- \( b \) is the bias term.

### Maximizing the Margin
The margin is defined as the distance between the closest data points of each class to the hyperplane. These closest points are known as the support vectors. The SVM aims to maximize this margin to improve the model's generalization ability. Mathematically, the margin is given by the formula:
\[ \frac{2}{\|w\|} \]
The goal is to maximize \( \frac{2}{\|w\|} \), which is equivalent to minimizing \( \|w\| \) or \( \|w\|^2 \) (for simpler computation).

### Constraints for Correct Classification
For data points \( (x_i, y_i) \) where \( y_i \) is either +1 or -1 (representing two classes), the constraints ensuring that these points are correctly classified by the hyperplane are:
\[ y_i (w \cdot x_i + b) \geq 1, \; \forall i \]
These constraints assert that:
- For \( y_i = 1 \), the data points are on one side of the hyperplane: \( w \cdot x_i + b \geq 1 \).
- For \( y_i = -1 \), the data points are on the other side: \( w \cdot x_i + b \leq -1 \).

### Soft Margin and Slack Variables
To allow for some misclassifications (especially in non-linearly separable cases), slack variables \( \xi_i \) are introduced. This leads to a soft margin SVM, which modifies the constraints to:
\[ y_i (w \cdot x_i + b) \geq 1 - \xi_i, \; \forall i \]
The objective function also adjusts to incorporate these slack variables:
\[ \min \frac{1}{2} \|w\|^2 + C \sum_{i=1}^n \xi_i \]
Where:
- \( C \) is a regularization parameter that controls the trade-off between maximizing the margin and minimizing the classification error.
- \( \xi_i \) are slack variables penalized in the objective, ensuring that points are either on the correct side of the margin or close to it.

### Quadratic Programming Solution
The linear SVM problem is typically solved using quadratic programming techniques because the objective function is quadratic and the constraints are linear. The solution involves finding the values of \( w \) and \( b \) that minimize the objective function while satisfying the constraints.

### Dual Problem
Often, it's beneficial to solve the SVM in its dual form, especially when extending SVMs to non-linear kernels. The dual formulation involves Lagrange multipliers \( \alpha_i \), and the problem transforms into:
\[ \max_\alpha \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i,j} \alpha_i \alpha_j y_i y_j x_i \cdot x_j \]
Subject to:
\[ \sum_{i=1}^n \alpha_i y_i = 0 \]
\[ 0 \leq \alpha_i \leq C, \; \forall i \]
Where \( \alpha_i \) are the Lagrange multipliers for each constraint. This form is particularly useful as it naturally accommodates the kernel trick for non-linear classification.

This dual formulation not only allows for solving the SVM problem efficiently but also extends the algorithm to handle non-linear boundaries by simply replacing the dot product with a kernel function, facilitating non-linear classification in higher-dimensional spaces through implicit mapping.