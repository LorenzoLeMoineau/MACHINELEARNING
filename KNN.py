import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, roc_curve, auc
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
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
        valid_axes_2d = []
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
                precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

                fpr, tpr, _ = roc_curve(y_test, knn.predict_proba(X_test)[:, 1], pos_label=1)
                roc_auc = auc(fpr, tpr)

                if accuracy >= accuracy_threshold:
                    print(f"\nResults for features: {visual_features}")
                    print("Classification Report:")
                    print(classification_report(y_test, y_pred))
                    print(f"Accuracy Score: {accuracy}")
                    print(f"Precision: {precision}, Recall: {recall}, F1-Score: {f1}")
                    valid_axes_2d.append((visual_features, X, y, knn, accuracy, precision, recall, f1, roc_auc))

        fig_2d, axes_2d = plt.subplots(len(valid_axes_2d), 1, figsize=(15, 10 * len(valid_axes_2d)))
        if len(valid_axes_2d) == 1:
            axes_2d = [axes_2d]

        for idx, (visual_features, X, y, knn, accuracy, precision, recall, f1, roc_auc) in enumerate(valid_axes_2d):
            x_min, x_max = X[visual_features[0]].min() - 1, X[visual_features[0]].max() + 1
            y_min, y_max = X[visual_features[1]].min() - 1, X[visual_features[1]].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                                 np.arange(y_min, y_max, 0.01))

            Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            cmap_background = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
            cmap_points = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
            axes_2d[idx].contourf(xx, yy, Z, alpha=0.8, cmap=cmap_background)
            scatter = axes_2d[idx].scatter(X[visual_features[0]], X[visual_features[1]],
                                            c=y, cmap=cmap_points, edgecolor='k', s=20)

            axes_2d[idx].set_title(f"{visual_features[0]} vs {visual_features[1]}")
            axes_2d[idx].set_xlabel(visual_features[0])
            axes_2d[idx].set_ylabel(visual_features[1])

            axes_2d[idx].legend(*scatter.legend_elements(), title="Classes")

            # Information inside the plot (no more outside)
            axes_2d[idx].text(1.05, 1, f"Accuracy: {accuracy:.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}\nF1-Score: {f1:.2f}\nAUC: {roc_auc:.2f}",
                              fontsize=10, ha='left', va='top', transform=axes_2d[idx].transAxes)

        plt.tight_layout()
        plt.subplots_adjust(hspace=1)  # Increase space between subplots vertically
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
                    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

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

                        # Information shifted to the right
                        ax.text2D(1.1, 0.5, f"Accuracy: {accuracy:.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}\nF1-Score: {f1:.2f}",
                                  transform=ax.transAxes, fontsize=10, verticalalignment='center', ha='left')

                        plot_idx_3d += 1

        plt.tight_layout()
        plt.subplots_adjust(hspace=1.2, wspace=0.6)  # Adjust vertical and horizontal space between subplots
        plt.show()
