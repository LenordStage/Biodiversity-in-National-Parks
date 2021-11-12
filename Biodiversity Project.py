import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from pandas import read_csv, unique, merge, options
import pandas as pd

df = read_csv("observations.csv")
df2 = read_csv("species_info.csv")
options.display.float_format = '{:,.0f}'.format
df.replace({'park_name': {'Great Smoky Mountains National Park': 'GSMNP'}}, inplace=True)
df.replace({'park_name': {'Yosemite National Park': 'YNP'}}, inplace=True)
df.replace({'park_name': {'Bryce National Park': 'BNP'}}, inplace=True)
df.replace({'park_name': {'Yellowstone National Park': 'YeNP'}}, inplace=True)
observations = df.groupby(['park_name']).observations.mean()
unique_values = df['scientific_name'].unique()
nunique_values = df['scientific_name'].nunique(dropna=True)
sns.barplot(data=df, x='park_name', y='observations')
plt.xlabel('Park Names')
plt.ylabel('Number of Observations')
plt.legend()
plt.show()
names = df['park_name'].unique()
names2 = unique(df.park_name)
# print(df.head())
# print(names2)
# print(observations)
# print(unique_values)
# print(nunique_values)
#print(df2.columns)
del df2['conservation_status']
#print(df2.columns)
bio = merge(df, df2)
# print(bio.columns)
# print(df2.columns)
# print(df.columns)
# print(bio)
bio.to_csv('bio.csv')
