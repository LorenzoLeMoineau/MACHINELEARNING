import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D

file_name = "burger-king-menu.csv"

encodings_to_try = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']
df = None

for encoding in encodings_to_try:
    try:
        df = pd.read_csv(file_name, encoding=encoding)
        print(f"File successfully loaded using {encoding} encoding:")
        break
    except UnicodeDecodeError:
        continue

if df is None:
    print("Error: Unable to read the file with the tested encodings.")
else:
    print("\nData preview:")
    print(df.head())
    print("\nData summary:")
    print(df.info())
    print("\nDescriptive statistics:")
    print(df.describe())

    selected_columns = ['Calories', 'Fat (g)', 'Sodium (mg)', 'Sugars (g)', 'Protein (g)', 'Cholesterol (mg)']
    target_column = 'Category'

    if any(col not in df.columns for col in selected_columns) or target_column not in df.columns:
        print("Required columns for KNN analysis are missing.")
    else:
        print("\nDetected columns for KNN analysis:", selected_columns)

        df = df.dropna(subset=selected_columns)
        df = df.dropna(subset=[target_column])

        label_encoder = LabelEncoder()
        df[target_column] = label_encoder.fit_transform(df[target_column])

        scaler = StandardScaler()
        df[selected_columns] = scaler.fit_transform(df[selected_columns])

        accuracy_threshold = 0.80

        print("\nGenerating 2D plots...")
        fig_2d, axes_2d = plt.subplots(len(selected_columns) - 1, len(selected_columns) - 1, figsize=(20, 15))
        fig_2d.subplots_adjust(hspace=0.4, wspace=0.4)
        axes_2d = axes_2d.ravel()

        plot_idx_2d = 0
        for i in range(len(selected_columns) - 1):
            for j in range(i + 1, len(selected_columns)):
                visual_features = [selected_columns[i], selected_columns[j]]
                X = df[visual_features]
                y = df[target_column]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                knn = KNeighborsClassifier(n_neighbors=5)
                knn.fit(X_train, y_train)

                y_pred = knn.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)

                if accuracy >= accuracy_threshold:
                    print(f"\nResults for features: {visual_features}")
                    print("Classification Report:")
                    print(classification_report(y_test, y_pred))
                    print("Accuracy Score:", accuracy)

                    x_min, x_max = X[visual_features[0]].min() - 1, X[visual_features[0]].max() + 1
                    y_min, y_max = X[visual_features[1]].min() - 1, X[visual_features[1]].max() + 1
                    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                                         np.arange(y_min, y_max, 0.01))

                    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
                    Z = Z.reshape(xx.shape)

                    cmap_background = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
                    cmap_points = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
                    axes_2d[plot_idx_2d].contourf(xx, yy, Z, alpha=0.8, cmap=cmap_background)
                    scatter = axes_2d[plot_idx_2d].scatter(X[visual_features[0]], X[visual_features[1]],
                                                           c=y, cmap=cmap_points, edgecolor='k', s=20)

                    axes_2d[plot_idx_2d].set_title(f"{visual_features[0]} vs {visual_features[1]}")
                    axes_2d[plot_idx_2d].set_xlabel(visual_features[0])
                    axes_2d[plot_idx_2d].set_ylabel(visual_features[1])
                    plot_idx_2d += 1

        plt.show()

        print("\nGenerating 3D plots...")
        plot_idx_3d = 0
        fig_3d = plt.figure(figsize=(20, 15))

        for i in range(len(selected_columns) - 2):
            for j in range(i + 1, len(selected_columns) - 1):
                for k in range(j + 1, len(selected_columns)):
                    visual_features = [selected_columns[i], selected_columns[j], selected_columns[k]]
                    X = df[visual_features]
                    y = df[target_column]

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    knn = KNeighborsClassifier(n_neighbors=5)
                    knn.fit(X_train, y_train)

                    y_pred = knn.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)

                    if accuracy >= accuracy_threshold:
                        print(f"\nResults for features: {visual_features}")
                        print("Classification Report:")
                        print(classification_report(y_test, y_pred))
                        print("Accuracy Score:", accuracy)

                        if plot_idx_3d % 9 == 0 and plot_idx_3d > 0:
                            plt.show()
                            fig_3d = plt.figure(figsize=(20, 15))

                        ax = fig_3d.add_subplot(3, 3, (plot_idx_3d % 9) + 1, projection='3d')
                        scatter = ax.scatter(X[visual_features[0]], X[visual_features[1]], X[visual_features[2]],
                                             c=y, cmap=ListedColormap(['#FF0000', '#00FF00', '#0000FF']),
                                             edgecolor='k', s=40)

                        ax.set_title(f"{visual_features[0]} vs {visual_features[1]} vs {visual_features[2]}")
                        ax.set_xlabel(visual_features[0])
                        ax.set_ylabel(visual_features[1])
                        ax.set_zlabel(visual_features[2])
                        plot_idx_3d += 1

        plt.show()
