import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy import stats
from sklearn.preprocessing import OneHotEncoder
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp


sns.set()

data = pd.read_csv(os.path.join("Data", "cleaned_villagers.csv"))

# ANOVA on ranking of the 4 most common species
# Cat, Rabbit, Squirrel, Frog

# Plot to see if they're normal

# data_plot = data[(data['Species'] == 'Cat') | (data['Species'] == 'Rabbit') |
#                  (data['Species'] == 'Squirrel') | (data['Species'] == 'Frog')]
# data_plot.hist('Rank', by='Species')
# plt.show()
# They don't look so normal, and we don't have that many data points to begin with, what now?

# Try ANOVA on Personalities
data.hist('Rank', by='Personality')
plt.show()

# More data points but they don't look normal either

# Try one way anova
data_plot = data.pivot(columns='Personality', values='Rank')
cranky = data_plot['Cranky'].dropna()
jock = data_plot['Jock'].dropna()
lazy = data_plot['Lazy'].dropna()
normal = data_plot['Normal'].dropna()
peppy = data_plot['Peppy'].dropna()
sisterly = data_plot['Sisterly'].dropna()
smug = data_plot['Smug'].dropna()
snooty = data_plot['Snooty'].dropna()
# print(cranky)
# Count how many sample we have for each group
# print(data.groupby('Personality').size())

# Sample size for Sisterly is 26, too low to use Central Limit Theorem
# ANOVA_pvalue = stats.f_oneway(cranky, jock, lazy,
#                      normal, peppy, sisterly,
#                      smug, snooty).pvalue
# print(f"ANOVA p-value: {ANOVA_pvalue}")

# Try Kruskal–Wallis test (Mann-Whitney U test for multiple groups)
Kruskal_pvalue = stats.kruskal(cranky, jock, lazy,
                     normal, peppy, sisterly,
                     smug, snooty).pvalue
print(f"Kruskal p-value for personality vs rank: {Kruskal_pvalue}\n")

# Dunn test for pairwise comparision
data_dunn = pd.melt(data_plot).dropna()
p_values = sp.posthoc_dunn(a=data_dunn, val_col='value', group_col='Personality', p_adjust='holm')
print("Dunn test p-values:")
print(p_values)
print(f"\n")
# At alpha = 0.05, there is a difference between Lazy & Cranky,  Normal vs Cranky, Peppy vs Cranky, Siserly vs Cranky,
# Snooty vs Lazy, Snooty vs Normal


# Hobbies vs Ranking
data_plot = data.pivot(columns='Hobbies', values='Rank')
nature = data_plot['Nature'].dropna()
fitness = data_plot['Fitness'].dropna()
play = data_plot['Play'].dropna()
education = data_plot['Education'].dropna()
fashion = data_plot['Fashion'].dropna()
music = data_plot['Music'].dropna()

# Count how many sample we have for each group
print("Size of each hobby group:")
print(data.groupby('Hobbies').size())
data.hist('Rank', by='Hobbies', bins=15)
# plt.show()
plt.savefig(os.path.join("Plots", "hist_hobbies_rank.png"))

ANOVA_pvalue = stats.f_oneway(nature, fitness, play, education, fashion, music).pvalue
print(f"\nHobbies vs Ranikng ANOVA p-value: {ANOVA_pvalue} \n")

#Tukey test
data_tukey = pd.melt(data_plot).dropna()
posthoc = pairwise_tukeyhsd(
    data_tukey['value'], data_tukey['Hobbies'],
    alpha=0.05)
print(posthoc)
fig = posthoc.plot_simultaneous()
# plt.show()
plt.savefig(os.path.join("Plots", "Tukey_hobbies_rank.png"))
# Kruskal_pvalue = stats.kruskal(nature, fitness, play, education, fashion, music).pvalue
# print(f"Kruskal p-value: {Kruskal_pvalue}")

# Gender vs Ranking
plt.ylabel('Count')
data.hist('Rank', by='Gender')
# plt.title("Histogram of villager ranking, grouped by gender")
# plt.show()
plt.savefig(os.path.join("Plots", "hist_gender_rank.png"))

female = data[data['Gender'] == 'F']['Rank']
male = data[data['Gender'] == 'M']['Rank']
print("\np-value for female rank normality test:", stats.normaltest(female).pvalue)
print("p-value for male rank normality test:", stats.normaltest(male).pvalue)

# Has more than 40 data points in each group but the histogram looks FAR from being normal
print("Size of each gender group:")
print(data.groupby('Gender').size())
# -> Use Mann Whitney U
print("p-value for M vs F ranking Mann Whitney U test:",stats.mannwhitneyu(female, male).pvalue) #0.199967 Fail to reject NULL hypothesis, gender doesn't affect ranking



