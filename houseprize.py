import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("C:\\Users\\rithu\\OneDrive\\Pictures\\Documents\\house-price-prediction-master\\house-price-prediction-master\\kc_house_data.csv")

# Show first 5 rows
print(data.head())

# Show column names
print("\nColumns in Dataset:")
print(data.columns)

# Remove missing values
data = data.dropna()

# Select last column as target
target_column = data.columns[-1]

print("\nTarget Column:", target_column)

# Features (input)
X = data.drop(target_column, axis=1)

# Target (output)
y = data[target_column]

# Convert text columns into numbers
X = pd.get_dummies(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = XGBRegressor()

# Train model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Calculate error
mse = mean_squared_error(y_test, predictions)

print("\nModel Trained Successfully!")
print("Mean Squared Error:", mse)

# Graph
plt.figure(figsize=(8,5))
plt.scatter(y_test, predictions)

plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("House Price Prediction")

plt.show()