import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
import pickle
import joblib

# Set a random seed for reproducibility
random_seed = 42

# Load the dataset
data = pd.read_csv("Real_Estate_Sales_2001-2020_GL.csv")

# Drop rows with missing values in 'Town' and 'Residential Type'
data.dropna(subset=['Town', 'Residential Type'], inplace=True)

# Sample a subset of the data for faster testing (optional)
data = data.sample(n=300000, random_state=random_seed)

# Extract features and target
X = data[['Town', 'Residential Type']].values
y = data['Sale Amount'].values

# One-hot encode categorical features
encoder = OneHotEncoder(sparse=False)
X = encoder.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

# Create and train a Decision Tree model with increased depth
# You can adjust the max_depth parameter to control the depth of the tree
max_depth = 500  # Adjust this value as needed
decision_tree = DecisionTreeRegressor(max_depth=max_depth, random_state=random_seed)
decision_tree.fit(X_train, y_train)

# Predict house prices on the testing data
y_pred = decision_tree.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print the results
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R²): {r2}")

filename = 'house_tree.pkl'
# Serialize the model
pickle.dump(decision_tree, open(filename, 'wb'))
# Serialize the ColumnTransformer
joblib.dump(encoder, 'encoder.pkl')
