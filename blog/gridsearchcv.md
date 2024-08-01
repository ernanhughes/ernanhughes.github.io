**GridSearchCV: A Comprehensive Guide**

`GridSearchCV` is a powerful tool in scikit-learn that enables users to perform hyperparameter tuning for
machine learning models using a grid-based search approach. In this section, we'll delve into the details of
`GridSearchCV`, its usage, and provide examples.

**What is GridSearchCV?**

`GridSearchCV` is a class that performs a brute-force search over a specified range of values for each
hyperparameter in a given model, using cross-validation to evaluate the performance of each combination. It's
particularly useful when:

1. You have a limited understanding of the optimal hyperparameters for your model.
2. You want to explore the entire hyperparameter space without relying on heuristics or manual tuning.

**How does GridSearchCV work?**

Here's a step-by-step explanation of how `GridSearchCV` works:

1. **Model preparation**: You create an instance of the desired machine learning model (e.g.,
`RandomForestClassifier`) and pass it to `GridSearchCV`.
2. **Hyperparameter space definition**: You specify a grid of hyperparameters for each parameter in the model
using the `param_grid` attribute. This defines the range of values that will be searched.
3. **Cross-validation**: `GridSearchCV` uses cross-validation (default is 5-fold CV) to evaluate the
performance of each combination of hyperparameters on the training data.
4. **Grid search**: The algorithm iterates through the grid of hyperparameters, fitting the model with each
combination and evaluating its performance using the cross-validated metric (e.g., accuracy).
5. **Best parameters identification**: After iterating through all combinations, `GridSearchCV` identifies the
set of hyperparameters that resulted in the best-performing model.

**Example usage**

Let's consider a simple example where we want to tune the `RandomForestClassifier` using `GridSearchCV`. We'll
explore different combinations of hyperparameters for the number of estimators (`n_estimators`) and the
maximum depth (`max_depth`).
```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Define model and hyperparameter space
model = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 5, 10]
}

# Initialize GridSearchCV instance
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')

# Perform grid search
grid_search.fit(X, y)

# Get the best-performing model and its hyperparameters
best_model = grid_search.best_estimator_
print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

# Print the scores for each combination of hyperparameters
for params, mean_score in zip(grid_search.cv_results_['params'], grid_search.cv_results_['mean_test_score']):
    print(f"Params: {params}, Mean test score: {mean_score}")
```
In this example, we define a `RandomForestClassifier` instance and specify a grid of hyperparameters for the
number of estimators (`n_estimators`) and the maximum depth (`max_depth`). We then initialize a `GridSearchCV`
instance with the model and hyperparameter space, perform the grid search using 5-fold cross-validation, and
retrieve the best-performing model and its hyperparameters.

**Conclusion**

`GridSearchCV` provides an efficient way to perform hyperparameter tuning for machine learning models using a
grid-based search approach. By following the steps outlined in this guide, you can leverage `GridSearchCV` to
explore different combinations of hyperparameters, identify the optimal configuration for your model, and
improve its performance on unseen data.