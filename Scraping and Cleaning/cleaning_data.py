import pandas as pd
import numpy as np
import re
import sys
import os

# Function to clean and format birthday strings
def clean_date_string(date_str):
    if pd.isnull(date_str):
        return None
    # Remove suffixes like "st", "nd", "rd", "th" only when they appear at the end
    date_str = re.sub(r'(st|nd|rd|th)$', '', date_str)
    return date_str.strip()

def cleaned_name_string(name_str):
    if pd.isnull(name_str):
        return None

    # Remove space and special characters
    name_str = re.sub(r'\W+', '', name_str)

    name_str = np.where(name_str == 'SporkNACracklePAL', 'Spork', name_str)
    name_str = np.where(name_str == 'CrackleSpork', 'Spork', name_str)
    name_str = np.where(name_str == 'JacobNAJakeyPAL', 'Jacob', name_str)
    name_str = np.where(name_str == 'BuckBrows', 'Buck', name_str)

    return name_str

def main():
    # Load the data
    villagers = pd.read_csv(os.path.join("Data", "villagers.csv"))
    ranking = pd.read_csv(os.path.join("Data", "ranking.csv"))

    # Clean up the Birthday column
    villagers['Birthday'] = villagers['Birthday'].apply(clean_date_string)

    # Parse the Birthday column into datetime
    villagers['Birthday'] = pd.to_datetime(villagers['Birthday'], format='%B %d', errors='coerce')

    # Drop the columns
    cleaned = villagers.drop(columns=['Image URL', 'Catchphrase'], axis=1)

    # Extract only the month and day
    cleaned['Birthday'] = cleaned['Birthday'].dt.strftime('%m-%d')

    # Convert 'Birthday' to datetime and extract the month
    cleaned['datetotime'] = pd.to_datetime(cleaned['Birthday'], format='%m-%d', errors='coerce')
    cleaned['BirthMonth'] = cleaned['datetotime'].dt.month
    cleaned = cleaned.drop(columns='datetotime')

    # Splitting Personality Column
    cleaned[['Gender', 'Personality']] = villagers['Personality'].str.split(n=1, expand=True)
    cleaned['Gender'] = cleaned['Gender'].str.split().str[-1]

    # Change the gender symbol to a word
    cleaned['Gender'] = np.where(cleaned["Gender"] == "♂", "F", "M")

    # Change the name of some villagers to match with the ranking data
    cleaned['Name'] = cleaned['Name'].apply(cleaned_name_string)

    # Change the name of villagers in the ranking data to match with the villagers data
    ranking['Name'] = ranking['Name'].apply(cleaned_name_string)

    # Join villagers data to ranking data
    cleaned = cleaned.set_index('Name')
    ranking = ranking.set_index('Name')
    cleaned = cleaned.join(ranking).reset_index()
    cleaned['Rank'] = cleaned['Rank'].fillna(0).astype(int) #remove decimal points from 'Rank'


    output_file = os.path.join("Data", "cleaned_villagers.csv")  
    cleaned.to_csv(output_file, index=False)


    print(f"Cleaned data saved to {output_file}")

if __name__ == '__main__':
    main()
