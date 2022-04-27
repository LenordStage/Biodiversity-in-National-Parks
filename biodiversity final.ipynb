#import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import chi2_contingency

df = pd.read_csv("observations.csv")
df2 = pd.read_csv("species_info.csv")
df2 = df2.drop_duplicates()
df = df.drop_duplicates()
###Names were too long for the graphs to work well
df2.replace({'category': {'Vascular Plant': 'V. Plant'}}, inplace=True)
df2.replace({'category': {"Nonvascular Plant": 'nV. Plant'}}, inplace=True)
df.replace({'park_name': {'Great Smoky Mountains National Park': 'GSMNP'}}, inplace=True)
df.replace({'park_name': {'Yosemite National Park': 'YNP'}}, inplace=True)
df.replace({'park_name': {'Bryce National Park': 'BNP'}}, inplace=True)
df.replace({'park_name': {'Yellowstone National Park': 'YeNP'}}, inplace=True)
df2.fillna('No Intervention', inplace=True)
print(df2['category'].unique())
print(df['park_name'].unique())
print(df.head())
print(df2.head())
print(df.dtypes)
print(df2.dtypes)
print(df['scientific_name'].nunique())
print(df2['scientific_name'].nunique())
print(df['scientific_name'].nunique(dropna=True))
####################   slide 5
df_slide5a = df2.drop("common_names", axis='columns')
df_slide5b = df_slide5a.drop('conservation_status', axis='columns')
df_slide5b.to_csv('df_slide5b.csv')
df3 = pd.read_csv('df_slide5b.csv')
ef = pd.merge(df, df3, on = 'scientific_name', how = "inner")
ef = ef.drop('Unnamed: 0', axis='columns')
ef = ef.drop_duplicates()
ef.drop(ef[ef['category'] == 'nV. Plant'].index, inplace = True)
ef.drop(ef[ef['category']=='V. Plant'].index, inplace=True)
ef.to_csv('ef.csv')
ef_s6= ef.groupby(['category', 'park_name'])['observations'].sum().reset_index()
ef_s6.to_csv('ef_s6.csv')
al = sns.barplot(x = "park_name", y = 'observations', hue='category', data=ef_s6)
sns.set_palette("pastel")
sns.despine(fig=None, ax=None, top=True, right=True, left=False, bottom=False, offset=None, trim=False)
plt.xlabel('Park Name')
plt.ylabel('Number of Observations')
#plt.legend(loc='best', frameon=True, fontsize="small")
plt.title('Distribution of Most Observed Animal Category')
for p in al.patches:
    al.annotate(format(p.get_height(), '.0f'),
                   (p.get_x() + p.get_width() / 2,
                    p.get_height()), ha='center', va='center',
                   size=8, xytext=(0, 8),
                   textcoords='offset points')
plt.rcParams['figure.figsize'] = [10, 10]
plt.show()
plt.savefig('DistributionofMostObservedAnimalCategory.png')
###### scientific name of the most observed animal############
ff = pd.merge(df, df2, on = 'scientific_name', how = "outer")
ff.drop(ff[ff['category'] == 'nV. Plant'].index, inplace = True)
ff.drop(ff[ff['category']=='V. Plant'].index, inplace=True)
ff = ff.drop('conservation_status', axis='columns')
ff = ff.drop_duplicates(subset=['scientific_name','park_name','observations','category']) 
ff = ff.drop_duplicates()
### This line is where the same data for the scientific name is used for the common name
ff.to_csv('jf.csv')
ff = ff.drop('common_names', axis='columns')
ff = ff.drop('category', axis='columns')
ff = ff.groupby(['scientific_name', 'park_name']).sum().reset_index()
ff.to_csv('ff.csv')
fff = pd.read_csv('ff.csv')
gf = fff.groupby(['scientific_name']).sum().reset_index()
gf = gf.drop('Unnamed: 0', axis='columns')
gf = gf[['scientific_name', 'observations']].sort_values('observations', ascending=False).nlargest(9, 'observations')

gf.to_csv('gf.csv', index=False)

gf_short = gf.head(9)
gf_short = gf_short.assign(common_name=['Beaver', 'Cougar', 'Raccoon', 'Rock Dove', 'Bat', 'Dove', 'Trout', 'Grouse', 'Heron'])
gf.to_csv('gf.csv', index=False)
#print(gf_short)
ap = sns.barplot(x = "common_name", y = 'observations',hue = 'scientific_name', data=gf_short)
sns.set_palette("pastel")
sns.despine(fig=None, ax=None, top=True, right=True, left=False, bottom=False, offset=None, trim=False)
plt.xlabel('Common Animal Name')
plt.ylabel('Total Number of Observations')
#plt.legend(loc='best', frameon=True, fontsize="extra small")
plt.title('Most Observed Animal- Scientific Name')
for p in ap.patches:
    ap.annotate(format(p.get_height(), '.0f'),
                   (p.get_x() + p.get_width() / 2,
                    p.get_height()), ha='center', va='center',
                   size=8, xytext=(0, 8),
                   textcoords='offset points')
plt.rcParams['figure.figsize'] = [10, 10]
plt.savefig('MostObservedAnimal4ParksTotaledScientificName.png')
plt.show()
#############   Common Name of the Most Observed Animal   ############
####### Here I just cleaned up the common names section and grouped a few animals together for clarity. 
jf = pd.read_csv('jf.csv')
jf = jf.drop('Unnamed: 0', axis='columns')
jf['common_names'] = jf['common_names'].str.replace(' " ', '')
jf['common_names'] = jf['common_names'].str.replace('(', '')
jf['common_names'] = jf['common_names'].str.replace(')', '')
jf['common_names'] = jf['common_names'].str.replace('American','')
jf['common_names'] = jf['common_names'].str.replace('Northern','')
jf['common_names'] = jf['common_names'].str.replace('Southern','')
jf['common_names'] = jf['common_names'].str.replace('Eastern','')
jf['common_names'] = jf['common_names'].str.replace('Western','')
jf['common_names'] = jf['common_names'].str.replace('Black','')
jf['common_names'] = jf['common_names'].str.replace('Red','')
jf['common_names'] = jf['common_names'].str.replace('Common','')
jf['common_names'] = jf['common_names'].str.replace('Mountain','')
jf['common_names'] = jf['common_names'].str.replace('California','')
jf['common_names'] = jf['common_names'].str.replace('Myotis','Bat')
jf['common_names'] = jf['common_names'].str.replace('Bullfrog','Frog')
jf['common_names'] = jf['common_names'].str.replace('Treefrog','Frog')
jf['common_names'] = jf['common_names'].str.replace('Kingsnake','Snake')
jf['common_names'] = jf['common_names'].str.replace('Watersnake','Snake')
jf['common_names'] = jf['common_names'].str.replace('Whipsnake','Snake')
jf['common_names'] = jf['common_names'].str.replace('Nightsnake','Snake')
jf['common_names'] = jf['common_names'].str.replace('Rattlesnake','Snake')
jf['common_names'] = jf['common_names'].str.replace('Jackrabbit','Rabbit')
jf['common_names'] = jf['common_names'].str.replace('Cottontail','Rabbit')
jf['common_names'] = jf['common_names'].str.replace('Salamanders','Salamander')
jf['common_names'] = jf['common_names'].str.replace(',','')
jf['new_common_names'] = jf['common_names'].str.split().str[-1]
jf = jf.drop('scientific_name', axis='columns')
jf = jf.drop('category', axis='columns')
jf = jf.groupby(['new_common_names', 'park_name']).sum().reset_index()
jf = jf.groupby(['new_common_names']).sum().reset_index()
jf = jf[['new_common_names', 'observations']].sort_values('observations', ascending=False).nlargest(9, 'observations')
jf.to_csv('jf2.csv', index = False)
#################Graphing for common name observations top 9##slide 12
aq = sns.barplot(x = "new_common_names", y = 'observations', data=jf)
sns.set_palette("pastel")
sns.despine(fig=None, ax=None, top=True, right=True, left=False, bottom=False, offset=None, trim=False)
plt.xlabel('Common Animal Name')
plt.ylabel('Total Number of Observations')
#plt.legend(loc='best', frameon=True, fontsize="extra small")
plt.title('Most Observed Animal Common Name - All Four Parks Totaled')
for p in aq.patches:
    aq.annotate(format(p.get_height(), '.0f'),
                   (p.get_x() + p.get_width() / 2,
                    p.get_height()), ha='center', va='center',
                   size=10, xytext=(0, 10),
                   textcoords='offset points')
plt.rcParams['figure.figsize'] = [10, 10]
plt.savefig('MostObservedAnimalCommonName4ParksTotaled.png')
plt.show()
#################   slide 7
df2_5 = df2.drop(df2[df2['conservation_status']=='No Intervention'].index, inplace = False)
ay = sns.countplot(data=df2_5, x = 'conservation_status', hue = 'category')
sns.set_palette("pastel")
sns.despine(fig=None, ax=None, top=True, right=True, left=False, bottom=False, offset=None, trim=False)
plt.xlabel('Conservation  Status')
plt.ylabel('Number of Species')
plt.legend(loc='right', frameon=False, fontsize="medium")
plt.title('Distribution of Conservation Status for Species')
for p in ay.patches:
    ay.annotate(format(p.get_height(), '.0f'),
                   (p.get_x() + p.get_width() / 2,
                    p.get_height()), ha='center', va='center',
                   size=8, xytext=(0, 8),
                   textcoords='offset points')
plt.rcParams['figure.figsize'] = [10, 10]
plt.show()
plt.savefig("distributionconservationstatusspecies.png")

########################### slide 4
observationsx = df.groupby(['park_name']).sum()
observationsx.to_csv('observationsy.csv')
observationsxy = pd.read_csv('observationsy.csv')
print(observationsx)

datax = {'park_name': ['BNP', 'GSMNP', 'YeNP', 'YNP'], 'observations count': [576025, 431820, 1443562, 863332]}
whereobservs = pd.DataFrame(datax)
ah = sns.barplot(x = 'park_name', y='observations count', hue=None,  data=whereobservs)
sns.set_palette("pastel")
sns.despine(fig=None, ax=None, top=True, right=True, left=False, bottom=False, offset=None, trim=False)
plt.legend(labels=["GSMNP = Great Smoky Mountains National Park","YNP = Yosemite National Park", "BNP = Bryce National Park", 'YeNP = Yellowstone National Park' ])
ah.set(xlabel='Park Name Abbreviated', ylabel='Number of Observations Made', title='Observations in These National Parks ')
for p in ah.patches:
    ah.annotate(format(p.get_height(), '.0f'),
                   (p.get_x() + p.get_width() / 2,
                   p.get_height()), ha='center', va='center',
                   size=8, xytext=(0, 8),
                   textcoords='offset points')
plt.rcParams['figure.figsize'] = [10, 10]
plt.show()
plt.savefig("Observations in These National Parks.png")


###################  Slide 3 
df2.drop(['common_names', 'conservation_status'], axis=1, inplace=False)
species_numberx = df2.groupby('category').count()
print(species_numberx)
ac = sns.countplot( x = 'category', data=df2)
sns.set_palette("pastel")
sns.despine(fig=None, ax=None, top=True, right=True, left=False, bottom=False, offset=None, trim=False)
plt.xlabel('Category')
plt.ylabel('Number of Species')
plt.title('Number of Species in Each Category')
for p in ac.patches:
    ac.annotate(format(p.get_height(), '.0f'),
                   (p.get_x() + p.get_width() / 2,
                    p.get_height()), ha='center', va='center',
                   size=8, xytext=(0, 8),
                   textcoords='offset points')
plt.rcParams['figure.figsize'] = [10, 10]
plt.savefig("number of species in each category.png")
plt.show()
###########################  Slide 8

df2 = df2.drop(df2[df2['conservation_status']=='No Intervention'].index, inplace = False)
df2 = df2.drop(df2[df2['conservation_status']=='Species of Concern'].index, inplace = False)
df2 = df2.drop ('common_names', axis=1, inplace=False)
df2.to_csv('dfv.csv')
dfv = pd.read_csv('dfv.csv')
print(dfv.head())
protected_data = dfv.groupby(['conservation_status', 'category']).nunique()
protected_data.to_csv('protected_data.csv')
protected_data_forsns = pd.read_csv('protected_data.csv')
aj = sns.barplot(x ='conservation_status', y='scientific_name', hue='category', data = protected_data_forsns,  edgecolor="none")
sns.set_palette("pastel")
sns.despine(fig=None, ax=None, top=True, right=True, left=False, bottom=False, offset=None, trim=False)
plt.rcParams['figure.figsize'] = [10, 10]
aj.set(xlabel='Conservation Status', ylabel='Number of Species', title='What is Endangered '),
for p in aj.patches:
    aj.annotate(format(p.get_height(), '.0f'),
                   (p.get_x() + p.get_width() / 2,
                    p.get_height()), ha='center', va='center',
                   size=8, xytext=(0, 8),
                   textcoords='offset points')
plt.savefig("what is endangered.png")
plt.show()

##################chi squared question

dfchi = pd.read_csv("species_info.csv")
dfchi = dfchi.drop_duplicates()
dfchi.fillna('No Intervention', inplace=True)
dfchi.drop(dfchi[dfchi['conservation_status'] == 'Endangered'].index, inplace = True)
dfchi.drop(dfchi[dfchi['conservation_status'] == 'In Recovery'].index, inplace = True)
dfchi.drop(dfchi[dfchi['conservation_status'] == 'Threatened'].index, inplace = True)
dfchi.to_csv('dfchi.csv')
chidata = dfchi.groupby(['category'])['conservation_status'].count()
print(chidata)

dfchib = pd.read_csv("species_info.csv")
dfchib = dfchib.drop_duplicates()
dfchib.fillna('No Intervention', inplace=True)
dfchib.drop(dfchib[dfchib['conservation_status'] == 'No Intervention'].index, inplace = True)
dfchib.drop(dfchib[dfchib['conservation_status'] == 'Species of Concern'].index, inplace = True)
dfchib.to_csv('dfchib.csv')
chidatab = dfchib.groupby(['category'])['conservation_status'].count()
print(chidatab)
chidata_table = pd.merge(chidata, chidatab, on = 'category')
chidata_table.rename(columns = {'conservation_status_x':'Not_Protected', 'conservation_status_y':'Protected'}, inplace=True)
chidata_table['Percent_Protected'] = chidata_table.Protected /(chidata_table.Protected + chidata_table.Not_Protected)*100
chidata_table['Percent_Protected'] = chidata_table['Percent_Protected'].round(decimals=2)
chidata_table.to_csv('chidata_table.csv')

#The chidata_table was copied using a snipet tool in windows. The actual chart came from the variable explorer in Spyder5, the IDE I use. It just pops it out!

print(chidata_table)
contingency1 = [[10, 204],
              [7, 514]]

print(contingency1)
(chi2_contingency(contingency1))
contingency2 = [[10, 204],
               [3, 77]]

print(contingency2)
print(chi2_contingency(contingency2))














