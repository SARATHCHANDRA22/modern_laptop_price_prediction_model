import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from train import X_test,y_test
# Load model and encoder
model = joblib.load("model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Load dataset
df = pd.read_csv(r"C:\Users\saisa\OneDrive\Documents\Desktop\Training_ML project\dataset\Laptop_price.csv")

#  Encoding
df["Brand"] = label_encoder.transform(df["Brand"])

# Features and target
X = df.drop("Price", axis=1)
y = df["Price"]

# Prediction
y_pred = model.predict(X_test)

# Plot actual vs predicted prices
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual Price vs Predicted Price")
plt.show()

# outputs
print("\nActual Prices:")
print(y.head())

print("\nPredicted Prices:")
print(y_pred[:5])


# Evaluation
print("\nEvaluation Metrics:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
