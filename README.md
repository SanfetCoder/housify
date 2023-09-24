import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load your dataset (replace 'your_dataset.csv' with your data file)
data = pd.read_csv('Real_Estate_Sales_2001-2020_GL.csv')

# Get user input for the town and property details
your_town = input('Enter your town: ')
residential_type = input('Enter the Residential Type of the property: ')

# Filter the data for your town
your_town_data = data[data['Town'] == your_town]
your_town_data = your_town_data[data['Residential Type'].notna()]
print(your_town_data['Residential Type'].head())
your_resident_data = your_town_data[your_town_data['Residential Type'] == residential_type]

# Assume 'X' contains your independent variables (features) and 'y' contains the target variable
# For this simplified model, we'll only use a constant feature (intercept) to predict the sale amount
X = np.ones((len(your_town_data), 1))
y = your_town_data['Sale Amount']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Make predictions for the training and testing sets
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate Mean Squared Error (MSE) for training and testing sets
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

# Print the MSE for both training and testing sets
print(f'Training MSE: {mse_train:.2f}')
print(f'Testing MSE: {mse_test:.2f}')

# Create a scatter plot for the training data
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, alpha=0.5, label='Training Data')
plt.xlabel('List Year')
plt.ylabel('Sale Amount')
plt.title(f'Training Data: List Year vs. Sale Amount for {your_town}')
plt.legend()
plt.grid(True)

# Show the plot for training data
plt.show()

# Create a scatter plot for the testing data
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, alpha=0.5, label='Testing Data')
plt.xlabel('List Year')
plt.ylabel('Sale Amount')
plt.title(f'Testing Data: List Year vs. Sale Amount for {your_town}')
plt.legend()
plt.grid(True)

# Show the plot for testing data
plt.show()

UncleFame/UncleFame is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->
