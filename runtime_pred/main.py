# Import necessary libraries
import numpy as np
import optuna
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Generate synthetic dataset (you can replace this with your own dataset)
X, y = make_regression(n_samples=100, n_features=1, noise=0.1, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a GradientBoostingRegressor model
gb_regressor = GradientBoostingRegressor(
    n_estimators=100,     # Number of boosting stages (trees)
    learning_rate=0.1,    # Step size shrinkage to prevent overfitting
    max_depth=3,          # Maximum depth of individual trees
    random_state=42       # Random seed for reproducibility
)

# Fit the model on the training data
gb_regressor.fit(X_train, y_train)

# Make predictions on the test data
y_pred = gb_regressor.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print("Mean Squared Error:", mse)
print("R-squared:", r2)

# You can also plot the original data points and the predicted values
import matplotlib.pyplot as plt

plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.scatter(X_test, y_pred, color='red', label='Predicted')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
