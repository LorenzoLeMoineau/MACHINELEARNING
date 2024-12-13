import pandas as pd
from sklearn.linear_model import LinearRegression
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

else:
    print("\nThe selected columns do not exist in the data. Please check your choices.")
