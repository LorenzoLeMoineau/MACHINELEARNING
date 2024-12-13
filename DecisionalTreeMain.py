import pandas as pd
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

MAX_DEPTH = 4
MIN_SAMPLES_SPLIT = 2
MIN_SAMPLES_LEAF = 1
RANDOM_STATE = 42

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

print("\nChoose the variable to predict by entering its corresponding number:")
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


target = get_valid_input("\nEnter the number of the variable to predict: ")

if target in data.columns:
    data = data.iloc[1:]  # Exclure la première ligne de la base de données
    data = data.dropna()

    X = data[[col for col in options.values() if col != target]]
    y = data[target]

    model = DecisionTreeRegressor(
        random_state=RANDOM_STATE,
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF
    )
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

    print("\nPerformance scores:")
    print(f"Mean Squared Error (MSE): {mse} ({mse_quality})")
    print(f"Root Mean Squared Error (RMSE): {rmse} ({rmse_quality})")
    print(f"Mean Absolute Error (MAE): {mae} ({mae_quality})")
    print(f"R² Score: {r2} ({r2_quality})")

    print("\nDecision Tree Visualization:")
    plt.figure(figsize=(20, 10))
    plot_tree(
        model,
        feature_names=X.columns,
        filled=True,
        rounded=True,
        fontsize=10
    )
    plt.title(f"Decision Tree for predicting '{target}'")
    plt.show()

    print("\nPredict the value for a product by providing input values:")
    user_input = {}
    for feature in X.columns:
        user_input[feature] = float(input(f"Enter the value for {feature}: "))

    input_data = pd.DataFrame([user_input])
    predicted_value = model.predict(input_data)[0]
    print(f"\nThe predicted value for '{target}' is: {predicted_value:.2f}")

else:
    print("\nThe selected column does not exist in the data. Please check your choices.")
