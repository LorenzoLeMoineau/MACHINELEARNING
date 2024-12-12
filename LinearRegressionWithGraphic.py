import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

print("Loading the CSV file containing the data.")
file_name = "burger-king-menu.csv"
data = pd.read_csv(file_name, encoding='latin1')

print("\nHere is an overview of the columns available in your data:")
print(data.columns)

options = {
    1: 'Sugars (g)',
    2: 'Sodium (mg)',
    3: 'Fat (g)',
    4: 'Calories',
    5: 'Protein (g)',
    6: 'Cholesterol (mg)'
}

print("\nChoose the independent and dependent variables by entering the corresponding numbers:")
for key, value in options.items():
    print(f"{key} : {value}")

def get_valid_input(prompt):
    while True:
        try:
            choice = int(input(prompt))
            if choice in options:
                return options[choice]
            else:
                print("Please enter a valid number (between 1 and 6).")
        except ValueError:
            print("Invalid input. Please enter a number.")

feature_1 = get_valid_input("\nEnter the number for the first independent variable: ")
feature_2 = get_valid_input("Enter the number for the second independent variable: ")
target = get_valid_input("Enter the number for the dependent variable: ")

if all(col in data.columns for col in [feature_1, feature_2, target]):
    data = data[[feature_1, feature_2, target]].dropna()

    X = data[[feature_1, feature_2]]
    y = data[target]

    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)

    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    def evaluate_metric(metric, thresholds, higher_is_better=True):
        if higher_is_better:
            if metric >= thresholds[0]:
                return "Excellent"
            elif metric >= thresholds[1]:
                return "Good"
            else:
                return "Poor"
        else:
            if metric <= thresholds[0]:
                return "Excellent"
            elif metric <= thresholds[1]:
                return "Good"
            else:
                return "Poor"

    mse_quality = evaluate_metric(mse, thresholds=[10, 50], higher_is_better=False)
    rmse_quality = evaluate_metric(rmse, thresholds=[5, 10], higher_is_better=False)
    mae_quality = evaluate_metric(mae, thresholds=[5, 10], higher_is_better=False)
    r2_quality = evaluate_metric(r2, thresholds=[0.75, 0.5], higher_is_better=True)

    print("\nLinear Regression Results:")
    print(f"Coefficients: {dict(zip(X.columns, model.coef_))}")
    print(f"Intercept: {model.intercept_}")
    print("\nPerformance Scores:")
    print(f"Mean Squared Error (MSE): {mse} ({mse_quality})")
    print(f"Root Mean Squared Error (RMSE): {rmse} ({rmse_quality})")
    print(f"Mean Absolute Error (MAE): {mae} ({mae_quality})")
    print(f"R² Score: {r2} ({r2_quality})")

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X[feature_1], X[feature_2], y, color='blue', label='Real Data')

    x_feature_1 = np.linspace(X[feature_1].min(), X[feature_1].max(), 10)
    x_feature_2 = np.linspace(X[feature_2].min(), X[feature_2].max(), 10)
    x_feature_1, x_feature_2 = np.meshgrid(x_feature_1, x_feature_2)
    y_pred_surface = (
        model.coef_[0] * x_feature_1 +
        model.coef_[1] * x_feature_2 +
        model.intercept_
    )
    ax.plot_surface(x_feature_1, x_feature_2, y_pred_surface, color='red', alpha=0.5)

    ax.set_title(f"Linear Regression: {target} as a function of {feature_1} and {feature_2}")
    ax.set_xlabel(feature_1)
    ax.set_ylabel(feature_2)
    ax.set_zlabel(target)
    plt.legend()
    plt.show()

    print("\nPredict the value for a product by providing input values:")
    user_input = {}
    user_input[feature_1] = float(input(f"Enter the value for {feature_1}: "))
    user_input[feature_2] = float(input(f"Enter the value for {feature_2}: "))

    input_data = pd.DataFrame([user_input])
    predicted_value = model.predict(input_data)[0]
    print(f"\nThe predicted value for '{target}' is: {predicted_value:.2f}")

else:
    print("\nThe selected columns do not exist in the data. Please check your choices.")