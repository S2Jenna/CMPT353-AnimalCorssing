import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

sns.set()

# Read the data
data = pd.read_csv(os.path.join("Data", "cleaned_villagers.csv"))

# box plot
plt.figure(1, figsize=(10, 10))
sns.boxplot(x=data['Species'], y=data['Rank'], data=data)
plt.xticks(rotation=90)
plt.title("Villager Popularity Ranking vs. Villager Species Box Plot")
# plt.show()
plt.savefig(os.path.join("Plots", "box_species_rank.png"))

# box plot but ordered by median
# https://stackoverflow.com/questions/21912634/how-can-i-sort-a-boxplot-in-pandas-by-the-median-values
plt.figure(2, figsize=(10, 10))
# grouped = data.groupby('Species')
# data2 = pd.DataFrame({col:vals['Rank'] for col,vals in grouped})
# med = data2.median()
# med.sort_values(inplace = True)
# data2 = data2[med.index]
# data2.boxplot()
# plt.xticks(rotation=90)
# # plt.show()
# plt.savefig(os.path.join("Plots", "box_species_rank_sorted.png"))

# box plot hobbies vs rank
sns.boxplot(x=data['Hobbies'], y=data['Rank'], data=data)
plt.xticks(rotation=90)
plt.title("Popularity Ranking vs. Hobbies Box Plot")
# plt.show()
plt.savefig(os.path.join("Plots", "box_hobbies_rank.png"))

# box plot Personality vs Ranking
plt.figure(3, figsize=(10, 10))
sns.boxplot(x=data['Personality'], y=data['Rank'], data=data)
plt.xticks(rotation=90)
plt.title("Popularity Ranking vs. Personality Box Plot")
# plt.show()
plt.savefig(os.path.join("Plots", "box_personality_rank.png"))

# box plot Gender vs Ranking
plt.figure(4, figsize=(6, 6))
sns.boxplot(x=data['Gender'], y=data['Rank'], data=data)
plt.xticks(rotation=90)
# plt.show()
plt.savefig(os.path.join("Plots", "box_gender_rank.png"))

# Bar plot of Hobbies vs Personalities
plt.figure(6, figsize=(15, 11))
contin = pd.crosstab(data['Personality'], data['Hobbies'])
ax = contin.plot(kind='bar', stacked=True, title='Bar plot')

# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# plt.show()
plt.savefig(os.path.join("Plots", "bar_personality_hobbies.png"))

# Histogram of all species count
plt.figure(8, figsize=(12, 10))
data['Species'].value_counts().plot(kind='bar')
plt.ylabel('Count')
# plt.show()
plt.savefig(os.path.join("Plots", "hist_species.png"))