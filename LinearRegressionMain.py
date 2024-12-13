import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

print("Loading the CSV file containing the data.")
file_name = "burger-king-menu.csv"
data = pd.read_csv(file_name, encoding='latin1')

# Exclure la première ligne de la base de données
data = data.iloc[1:]

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

print("\nChoose the variable to predict by entering a corresponding number:")
for key, value in options.items():
    print(f"{key} : {value}")

while True:
    try:
        choice = int(input("\nEnter a number between 1 and 6: "))
        if choice in options:
            target = options[choice]
            break
        else:
            print("Please enter a valid number (between 1 and 6).")
    except ValueError:
        print("Invalid input. Please enter a number.")

if target in data.columns:
    data = data.dropna()

    X = data[[col for col in options.values() if col != target]]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

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
    print(f"Predicted variable: {target}")
    print(f"Coefficients: {dict(zip(X.columns, model.coef_))}")
    print(f"Intercept: {model.intercept_}")
    print("\nPerformance Scores:")
    print(f"Mean Squared Error (MSE): {mse} ({mse_quality})")
    print(f"Root Mean Squared Error (RMSE): {rmse} ({rmse_quality})")
    print(f"Mean Absolute Error (MAE): {mae} ({mae_quality})")
    print(f"R² Score: {r2} ({r2_quality})")

    print("\nYou can now predict the value for a specific product.")
    user_input = {}
    for feature in X.columns:
        user_input[feature] = float(input(f"Enter the value for {feature}: "))

    input_data = pd.DataFrame([user_input])

    predicted_value = model.predict(input_data)[0]
    print(f"\nThe predicted value for '{target}' is: {predicted_value:.2f}")
else:
    print("\nThe selected column does not exist in the data. Please check your choices.")
