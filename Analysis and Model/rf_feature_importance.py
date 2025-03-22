import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_classif
from scipy import stats
import seaborn as sns


def prepare_data():
    mod_data = pd.read_csv(os.path.join("Data", "cat_to_num.csv"))

    # Split Data
    X = mod_data.drop(columns='Tier', axis=1).values
    y = mod_data['Tier'].values
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=42, stratify=y)

    return X_train, X_valid, y_train, y_valid, mod_data.columns


def build_random_forest(X_train, y_train):
    # Initialize the Random Forest model
    rf_model = RandomForestClassifier(random_state=42)

    # Hyperparameters for tuning
    param_grid = {
        "n_estimators": [100, 200, 400],
        "max_depth": [None, 5, 10],
        "min_samples_leaf": [1, 2, 5]
    }

    # Grid search for hyperparameter tuning
    grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Get the best model and its parameters
    best_rf_model = grid_search.best_estimator_
    #print(f"Best Hyperparameters: {grid_search.best_params_}")
    
    return best_rf_model


def plot_feature_importance(model, feature_names):
    # Get the feature importances
    importances = model.feature_importances_
    
    # Sort the importances in descending order
    indices = np.argsort(importances)[::-1]
    
    # Limit to top 10 most important features
    top_n = 10
    indices = indices[:top_n]

    # Get the top 10 feature names and importances
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    # Set the pastel color palette
    pastel_colors = sns.color_palette("pastel", n_colors=top_n)

    # Plot the feature importances
    plt.figure(figsize=(12, 8))
    plt.title("Top 10 Feature Importances")
    plt.barh(range(len(top_importances)), top_importances, align="center", color=pastel_colors)
    plt.yticks(range(len(top_importances)), top_features)
    plt.xlabel("Relative Importance")
    plt.savefig(os.path.join("Plots", "rf_importance.png"))
    plt.show()


def main():
    X_train, X_valid, y_train, y_valid, mod_data = prepare_data()
    rf_model = build_random_forest(X_train, y_train)
    plot_feature_importance(rf_model, mod_data)
    



if __name__ == '__main__':
    main()