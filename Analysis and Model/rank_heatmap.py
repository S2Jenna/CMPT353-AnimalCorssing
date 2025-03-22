import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import sys
import os

def main(infile):
    df = pd.read_csv(infile)

    # Convert 'Birthday' to datetime and extract the month
    df['datetotime'] = pd.to_datetime(df['Birthday'], format='%m-%d', errors='coerce')
    df['BirthMonth'] = df['datetotime'].dt.month
    df = df.drop(columns='datetotime')

    # Convert categorical columns to numeric using LabelEncoder
    label_encoder = LabelEncoder()
    df['Personality'] = label_encoder.fit_transform(df['Personality'])
    df['Species'] = label_encoder.fit_transform(df['Species'])
    df['Hobbies'] = label_encoder.fit_transform(df['Hobbies'])
    df['Gender'] = label_encoder.fit_transform(df['Gender'])

    res = df.drop(columns=['Name', 'Birthday'])

    # Compute correlation matrix
    correlation_matrix = res.corr()

    # Extract correlations of 'Rank' with other variables
    rank_correlations = correlation_matrix[['Rank']]

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(rank_correlations, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Rank vs. Other Variables")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    infile = os.path.join("Data", sys.argv[1])
    main(infile)
