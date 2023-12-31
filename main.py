from fastapi import FastAPI
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pickle
from fastapi.middleware.cors import CORSMiddleware
import joblib
import uvicorn

origins = [
    "http://localhost",
    "http://localhost:3000",
    "https://yourfrontenddomain.com",
    "http://localhost:5173",
    "https://subtle-bunny-7e34cc.netlify.app"
]

app = FastAPI()

app.add_middleware(
  CORSMiddleware,
  allow_origins=origins,
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"]
)

# Define the name of the saved model file
filename = "house_tree.pkl"

# de-serialize (load) the object
model = pickle.load(open(filename, 'rb'))

# Column Transformer
encoder = joblib.load('encoder.pkl')

@app.get('/{town}/{houseType}')
def root(town, houseType):
    estimated = predict_price(town, houseType)
    return {
        "predicted_price": estimated # Return the predicted value as a message
    }

def predict_price(town, houseType):

    # Custom input data (replace this with your values)
    custom_value = pd.DataFrame({'Town': [town], 'Residential Type': [houseType]})

    # Encode the custom input data
    custom_input_encoded = encoder.transform(custom_value)

    # Make predictions
    custom_output = model.predict(custom_input_encoded)

    predicted_price = custom_output[0]

    return predicted_price

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=80)