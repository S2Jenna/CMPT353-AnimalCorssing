import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

def prepare_data():
    data = pd.read_csv(os.path.join("Data", "cleaned_villagers.csv"))

    # Categorize the Rank
    bins = [0, 50, 150, 300, 375, 413]
    labels = ['Top', 'Mid', 'Average', 'Low', 'Bottom']
    data['Tier'] = pd.cut(data['Rank'], bins=bins, labels=labels, right=True)

    # Get BirthMonth
    # data['datetotime'] = pd.to_datetime(data['Birthday'], format='%m-%d', errors='coerce')
    # data['BirthMonth'] = data['datetotime'].dt.month
    # data = data.drop(columns='datetotime')

    # Convert categorical predictor variables to numerical
    mod_data = data.drop(columns=['Name', 'Birthday', 'Rank'])
    cat_columns = mod_data.select_dtypes(['object']).columns
    mod_data[cat_columns] = mod_data[cat_columns].apply(lambda x: pd.factorize(x)[0])

    # Split Data
    X = mod_data[['Personality', 'Species', 'Hobbies', 'Gender', 'BirthMonth']].values
    y = mod_data['Tier'].values
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=42, stratify=y)

    return X_train, X_valid, y_train, y_valid

def build_models(X_train, y_train):
    # Models
    models = {
        "Bayesian Classifier": make_pipeline(GaussianNB()),
        "kNN": KNeighborsClassifier(),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(random_state=42)),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

    # Hyperparameters for tuning
    param_grids = {
        "kNN": {"n_neighbors": [5, 10, 15, 20]},
        "Random Forest": {"n_estimators": [100, 200, 400], "max_depth": [None, 5, 10], "min_samples_leaf": [1, 2, 5]},
        "Logistic Regression": {"logisticregression__C": [0.01, 0.1, 1, 10]},
        "Gradient Boosting": {"n_estimators": [100, 200], "max_depth": [3, 5, 7], "learning_rate": [0.01, 0.1, 0.2]},
    }

    # Train and tune models
    tuned_models = {}
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        if model_name in param_grids:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            grid_search = GridSearchCV(model, param_grids[model_name], cv=cv, scoring='accuracy', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            tuned_models[model_name] = grid_search.best_estimator_
            print(f"Best params for {model_name}: {grid_search.best_params_}")
        else:
            model.fit(X_train, y_train)
            tuned_models[model_name] = model

    return tuned_models

def compute_accuracy(tuned_models, X_valid, y_valid):
    results = {}
    for model_name, model in tuned_models.items():
        accuracy = model.score(X_valid, y_valid)
        results[model_name] = accuracy
        print(f"{model_name} Accuracy: {accuracy:.3f}")

    # Find best model
    best_model_name = max(results, key=results.get)
    best_model_accuracy = results[best_model_name]
    print(f"\nBest Model: {best_model_name} with Accuracy: {best_model_accuracy:.3f}")

def main():
    X_train, X_valid, y_train, y_valid = prepare_data()
    tuned_models = build_models(X_train, y_train)
    compute_accuracy(tuned_models, X_valid, y_valid)

if __name__ == '__main__':
    main()
