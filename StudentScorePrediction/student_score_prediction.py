import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# 1. Load dataset
file_path = "StudentsPerformance.csv"  
df = pd.read_csv(file_path)

# 2. Check for missing values
print("Missing values:\n", df.isnull().sum())

# 3. Basic info
print("\nDataset Info:")
print(df.info())

# 4. Data visualization
plt.scatter(df['study_hours'], df['exam_scores'], color='blue')
plt.title("Study Hours vs Exam Scores")
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.grid(True)
plt.show()

# 5. Split data into features and target
X = df[['study_hours']]   # Feature
y = df['exam_scores']     # Target

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 8. Predictions
y_pred = model.predict(X_test)

# 9. Visualization of predictions
plt.scatter(X_test, y_test, color='blue', label="Actual")
plt.scatter(X_test, y_pred, color='red', label="Predicted")
plt.plot(X_test, y_pred, color='green')
plt.title("Actual vs Predicted Exam Scores")
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.legend()
plt.grid(True)
plt.show()

# 10. Model evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# 11. Bonus: Polynomial Regression (optional)
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split(
    X_poly, y, test_size=0.2, random_state=42
)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train_poly)
y_poly_pred = poly_model.predict(X_test_poly)

poly_r2 = r2_score(y_test_poly, y_poly_pred)

print(f"\nPolynomial Regression R² Score: {poly_r2:.2f}")
