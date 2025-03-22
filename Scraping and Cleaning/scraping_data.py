from bs4 import BeautifulSoup
import requests
import pandas as pd
import os
import csv

def main():

    # Ranking Data Scraping
    # URL to get ranking data
    ranking_url = 'https://www.animalcrossingportal.com/tier-lists/new-horizons/all-villagers/'

    # Send the request to the web page
    response = requests.get(ranking_url)

    # Check if the request was successful
    if response.status_code == 200:
        print("Ranking page retrieved successfully")

        # Use BeautifulSoup to get html code in lxml format
        #https://stackoverflow.com/questions/24398302/bs4-featurenotfound-couldnt-find-a-tree-builder-with-the-features-you-requeste
        soup = BeautifulSoup(response.text, 'lxml')

        # Function to get villager name and rank
        def name_extract(name, rank):
            return (name.getText(), rank.getText())

        # Extract data and write to a data frame
        names = soup.find_all(class_="c-candidate-name")
        ranks = soup.find_all(class_="c-candidate-rank")
        data = pd.DataFrame(list(map(name_extract, names, ranks)),
                                 columns=['Name', 'Rank'])

        # Save data to csv file
        output_file = os.path.join("Data", "ranking.csv")
        data.to_csv(output_file, index=False)

        print(f"Ranking data saved to {output_file}")

    else:
        print("Failed to retrieve the ranking page")


    # Villagers Data Scraping
    # URL to get villager data
    villager_url = 'https://animalcrossing.fandom.com/wiki/Villager_list_(New_Horizons)'

    # Send the request to the web page
    response = requests.get(villager_url)

    # Check if the request was successful
    if response.status_code == 200:
        print("Villagers page retrieved successfully")

        # Use BeautifulSoup to get html code in lxml format
        soup = BeautifulSoup(response.text, 'lxml')

        # Prepare to write to CSV
        output_file = os.path.join("Data", "villagers.csv")
        header = ["Name", "Image URL", "Personality", "Species", "Birthday", "Catchphrase", "Hobbies"]

        with open(output_file, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(header)

            total_villagers = 0
            # Find the table and extract its content
            table = soup.find('table', {'class': 'roundy sortable'})
            if table:
                rows = table.find('tbody').find_all('tr')
                for row in rows:
                    cells = row.find_all('td')
                    if cells:
                        name = cells[0].text.strip()
                        image = cells[1].find('img')['src'] if cells[1].find('img') else None
                        personality = cells[2].text.strip()
                        species = cells[3].text.strip()
                        birthday = cells[4].text.strip()
                        catchphrase = cells[5].text.strip()
                        hobbies = cells[6].text.strip()

                        # Write data to CSV
                        writer.writerow([name, image, personality, species, birthday, catchphrase, hobbies])
                        total_villagers += 1

                print(f"Villagers data saved to {output_file}")
            else:
                print("Villagers table not found.")
    else:
        print("Failed to retrieve the villagers page")


if __name__ == '__main__':
    main()