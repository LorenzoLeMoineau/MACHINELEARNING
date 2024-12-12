# Importer les bibliothèques nécessaires
from google.colab import files
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Télécharger le fichier
print("Veuillez télécharger votre fichier CSV contenant les données.")
uploaded = files.upload()

# Charger le fichier dans un DataFrame pandas
file_name = list(uploaded.keys())[0]
data = pd.read_csv(file_name, encoding='latin1')

# Afficher un aperçu des colonnes disponibles
print("\nVoici un aperçu des colonnes disponibles dans vos données :")
print(data.columns)

# Liste des options disponibles avec des numéros associés
options = {
    1: 'Sugars (g)',
    2: 'Sodium (mg)',
    3: 'Fat (g)',
    4: 'Calories',
    5: 'Protein (g)',
    6: 'Cholesterol (mg)'
}

# Afficher les options avec des numéros associés
print("\nChoisissez les variables indépendantes et dépendante en entrant les numéros correspondants :")
for key, value in options.items():
    print(f"{key} : {value}")

# Boucle pour garantir une entrée valide pour chaque sélection
def get_valid_input(prompt):
    while True:
        try:
            choice = int(input(prompt))
            if choice in options:
                return options[choice]
            else:
                print("Veuillez entrer un numéro valide (entre 1 et 6).")
        except ValueError:
            print("Entrée invalide. Veuillez entrer un numéro.")

# Sélection des variables
feature_1 = get_valid_input("\nEntrez le numéro pour la première variable indépendante : ")
feature_2 = get_valid_input("Entrez le numéro pour la deuxième variable indépendante : ")
target = get_valid_input("Entrez le numéro pour la variable dépendante : ")

# Vérifier si les colonnes choisies existent dans les données
if all(col in data.columns for col in [feature_1, feature_2, target]):
    # Nettoyer les données
    data = data[[feature_1, feature_2, target]].dropna()

    # Définir les variables indépendantes (features) et dépendante (target)
    X = data[[feature_1, feature_2]]
    y = data[target]

    # Créer le modèle de régression linéaire
    model = LinearRegression()
    model.fit(X, y)

    # Prédire les valeurs
    y_pred = model.predict(X)

    # Calculer les métriques
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    # Fonction pour évaluer la qualité des scores
    def evaluate_metric(metric, thresholds, higher_is_better=True):
        if higher_is_better:
            if metric >= thresholds[0]:
                return "Excellent"
            elif metric >= thresholds[1]:
                return "Bon"
            else:
                return "Faible"
        else:
            if metric <= thresholds[0]:
                return "Excellent"
            elif metric <= thresholds[1]:
                return "Bon"
            else:
                return "Faible"

    # Évaluation qualitative des métriques
    mse_quality = evaluate_metric(mse, thresholds=[10, 50], higher_is_better=False)
    rmse_quality = evaluate_metric(rmse, thresholds=[5, 10], higher_is_better=False)
    mae_quality = evaluate_metric(mae, thresholds=[5, 10], higher_is_better=False)
    r2_quality = evaluate_metric(r2, thresholds=[0.75, 0.5], higher_is_better=True)

    # Afficher les coefficients et métriques
    print("\nRésultats de la régression linéaire :")
    print(f"Coefficients : {dict(zip(X.columns, model.coef_))}")
    print(f"Intercept (ordonnée à l'origine) : {model.intercept_}")
    print("\nScores de performance :")
    print(f"Mean Squared Error (MSE) : {mse} ({mse_quality})")
    print(f"Root Mean Squared Error (RMSE) : {rmse} ({rmse_quality})")
    print(f"Mean Absolute Error (MAE) : {mae} ({mae_quality})")
    print(f"R² Score : {r2} ({r2_quality})")

    # Visualisation 3D
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Points réels
    ax.scatter(X[feature_1], X[feature_2], y, color='blue', label='Données réelles')

    # Surface de prédiction
    x_feature_1 = np.linspace(X[feature_1].min(), X[feature_1].max(), 10)
    x_feature_2 = np.linspace(X[feature_2].min(), X[feature_2].max(), 10)
    x_feature_1, x_feature_2 = np.meshgrid(x_feature_1, x_feature_2)
    y_pred_surface = (
        model.coef_[0] * x_feature_1 +
        model.coef_[1] * x_feature_2 +
        model.intercept_
    )
    ax.plot_surface(x_feature_1, x_feature_2, y_pred_surface, color='red', alpha=0.5)

    # Configurations de l'affichage
    ax.set_title(f"Régression linéaire : {target} en fonction de {feature_1} et {feature_2}")
    ax.set_xlabel(feature_1)
    ax.set_ylabel(feature_2)
    ax.set_zlabel(target)
    plt.legend()
    plt.show()

    # Fonctionnalité pour prédire la teneur d'un élément spécifique
    print("\nPrédisez la teneur pour un produit en fournissant des valeurs :")
    user_input = {}
    user_input[feature_1] = float(input(f"Entrez la valeur pour {feature_1} : "))
    user_input[feature_2] = float(input(f"Entrez la valeur pour {feature_2} : "))

    # Effectuer la prédiction
    input_data = pd.DataFrame([user_input])
    predicted_value = model.predict(input_data)[0]
    print(f"\nLa teneur prédite en '{target}' pour ce produit est : {predicted_value:.2f}")

else:
    print("\nLes colonnes choisies ne sont pas présentes dans le fichier. Veuillez vérifier vos choix.")
