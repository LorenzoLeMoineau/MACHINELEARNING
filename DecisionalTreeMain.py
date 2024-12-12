import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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
    data = data.dropna()
    X = data[[col for col in options.values() if col != target]]
    y = data[target]

    model = DecisionTreeRegressor(random_state=42, max_depth=4)
    model.fit(X, y)

    y_pred = model.predict(X)

    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print("\nPerformance scores:")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"RÂ² Score: {r2}")

    print("\nPredict the value for a product by providing input values:")
    user_input = {}
    for feature in X.columns:
        user_input[feature] = float(input(f"Enter the value for {feature}: "))

    input_data = pd.DataFrame([user_input])
    predicted_value = model.predict(input_data)[0]
    print(f"\nThe predicted value for '{target}' is: {predicted_value:.2f}")

else:
    print("\nThe selected column does not exist in the data. Please check your choices.")