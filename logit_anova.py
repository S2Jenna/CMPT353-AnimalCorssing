import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
import os
import matplotlib.pyplot as plt

def prepare_data():
    data = pd.read_csv(os.path.join("Data", "cat_to_num.csv"))

    X = data.drop(columns='Tier')
    y = data['Tier']
    
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, random_state=42, stratify=y
    )

    return X_train, X_valid, y_train, y_valid, X.columns

def fit_logistic_regression(X_train, y_train, feature_names):
    """
    Fits a Logistic Regression model and performs an ANOVA test to determine feature significance.
    Prints and returns a DataFrame with feature names and their p-values.
    """
    # Create a pipeline with scaling and logistic regression
    pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.01, random_state=42) # use the best parameter that we got from prediction_model_V2.py
    )
    
    # Fit the pipeline
    pipeline.fit(X_train, y_train)

    # Perform ANOVA test
    f_values, p_values = f_classif(X_train, y_train)

    # Create a DataFrame for ANOVA results
    anova_results = pd.DataFrame({
        "Feature Index": range(len(feature_names)),
        "Feature Name": feature_names,
        "p-value": p_values
    })

    # Sort the results by p-value
    anova_results = anova_results.sort_values("p-value").reset_index(drop=True)

    # Display results
    # print("ANOVA Test Results (sorted by p-value):")
    # print(anova_results)

    # Display the top 10 features with the lowest p-values
    print("\nTop 10 Features with Lowest p-values:")
    print(anova_results.head(10))

    return anova_results

def plot_anova_results(anova_results):
    """
    Plots the ANOVA results as a bar chart of p-values for all features.
    """
    plt.figure(figsize=(12, 6))
    plt.bar(anova_results["Feature Name"], anova_results["p-value"], color="skyblue")
    plt.xticks(rotation=90)
    plt.ylabel("p-value")
    plt.title("ANOVA Test Results: Feature p-values")
    plt.axhline(y=0.05, color='red', linestyle='--', label='Significance Threshold (p=0.05)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("Plots", "Logit_ANOVA.png"))
    plt.show()


def main():
    X_train, X_valid, y_train, y_valid, feature_names = prepare_data()
    anova_results = fit_logistic_regression(X_train, y_train, feature_names)
    plot_anova_results(anova_results)
    



if __name__ == '__main__':
    main()