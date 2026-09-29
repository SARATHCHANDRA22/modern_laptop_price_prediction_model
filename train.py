import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv(r"C:\Users\saisa\OneDrive\Documents\Desktop\Training_ML project\dataset\Laptop_price.csv")

# Display first 5 rows

print(df.head())

# dataset information
print(df.info())

#  statistical summary
print(df.describe())


#  input features and target
X = df.drop("Price", axis=1)
y = df["Price"]

label_encoder = LabelEncoder()

X["Brand"] = label_encoder.fit_transform(X["Brand"])



# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)



# Create the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

print("\nModel training completed!")
import joblib
joblib.dump(model, "model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
