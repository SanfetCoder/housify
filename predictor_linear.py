import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import joblib

# Dataset
data = pd.read_csv("Real_Estate_Sales_2001-2020_GL.csv")
# Drop None Column
data.dropna(subset=['Town', 'Residential Type'], inplace=True)
# Extract feature and target
X = data[['Town','Residential Type']].values
print(X.shape)
y = data['Sale Amount'].values

# Encode Categorical Data
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(sparse=False), [0,1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X.shape)

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Creating the model
regressor = LinearRegression()
# Training the model
regressor.fit(X_train, y_train)

# Predicted value
y_pred = regressor.predict(X_test)

# Evaluate the performance of the model
# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)

# Calculate R-squared (R²)
r2 = r2_score(y_test, y_pred)

# Print the evaluation metrics
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R²): {r2}")

filename = 'house_linear.pkl'
# Serialize the model
pickle.dump(regressor, open(filename, 'wb'))
# Serialize the ColumnTransformer
joblib.dump(ct, 'column_transformer.pkl')
