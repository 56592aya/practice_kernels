import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
from scipy.stats import ttest_ind
from scipy.stats import probplot
import pylab
import random
import seaborn as sns
import os 

print(os.listdir("input/"))
shoes = pd.read_csv("input/Datafiniti_Womens_Shoes.csv")

shoes.describe().transpose()
shoes.columns
shoes.dtypes

prices = shoes['prices.amountMax']
prices.describe().transpose()

plt.hist(prices, edgecolor="black", bins=30)
plt.title("some histogram")
plt.xlabel("prices")
plt.ylabel("counts")

####################
shoes['colors'].head(3)
shoes['colors'] = shoes['colors'].str.lower()
shoes['colors'].head(3)

probplot(shoes['prices.amountMax'], dist='norm', plot=pylab)
plt.show()



shoes.dropna(subset=['colors'], inplace=True)
shoes['pink'] = shoes['colors'].apply(lambda x: 'pink' in x)

a = shoes.loc[shoes['pink'] == True, 'prices.amountMax']
b = shoes.loc[shoes['pink'] == False, 'prices.amountMax']
print(ttest_ind(a, b, equal_var=False))

fig, ax = plt.subplots(1, 2)
sns.distplot(a, kde=False, ax = ax[0])
ax[0].set_xlabel('prices for pink')
ax[0].set_ylabel('freqs')
ax[0].set_title("pink prices")
sns.distplot(b, kde=False, ax = ax[1])
ax[1].set_xlabel('prices for other colors')
ax[1].set_ylabel('freqs')
ax[1].set_title("nonpink prices")

plt.show()

os.listdir("input/")
cereal = pd.read_csv("input/cereal.csv")
cereal.describe().transpose()
cereal.dtypes

cereal.sample(5)

type_freqs = cereal['type'].value_counts()
mfr_freqs = cereal['mfr'].value_counts()

type_ix = type_freqs.index
type_bar_positions = range(len(type_ix))
mfr_ix = mfr_freqs.index
mfr_bar_positions = range(len(mfr_ix))


# sns.barplot(type_ix, type_freqs.values)
plt.bar(type_ix, type_freqs.values)
plt.xticks(labels = type_ix, ticks=type_bar_positions)
sns.barplot(mfr_ix, mfr_freqs.values)

contingency_tab = pd.crosstab(cereal['type'], cereal['mfr'])
contingency_tab
scipy.stats.chi2_contingency(contingency_tab)

####################

permits = pd.read_csv("input/Building_Permits.csv")
permits.sample(5)

missing_value_counts = permits.isnull().sum()

total_values = np.product(permits.shape)
total_missing = np.sum(missing_value_counts)
print((total_missing/total_values)*100)

permits.dtypes
missing_value_counts
permits["Street Number Suffix"].sample(10)
permits.fillna({"Street Number Suffix":""}, inplace=True)
permits["Street Number Suffix"].sample(10)
permits[["Street Number Suffix"]] = permits[["Street Number Suffix"]].replace("", np.nan)
permits[["Street Number Suffix"]] = permits[["Street Number Suffix"]].replace(np.nan, "")
permits[["Street Number Suffix"]]

rows_with_na_dropped = permits.dropna(axis=0)
rows_with_na_dropped
columns_with_na_dropped = permits.dropna(axis=1)
columns_with_na_dropped

permits.sample(5).transpose()
missing_value_counts
permits.dtypes
permits.fillna(method='bfill', axis=0).fillna(0)

#some other stuff
permits.select_dtypes(exclude="object").columns
permits.drop(['Zipcode'], axis=1)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
##note the double bracket[expects 2d], and the .values(expects numpy ndarray)
x = permits[['Number of Existing Stories']].values
#does mean transform
transformed_x = imputer.fit_transform(x)
np.isnan(transformed_x).sum()


#Parsing Dates
from  mlxtend.preprocessing import  minmax_scaling
import datetime

earthquakes = pd.read_csv("input/earthquakes.csv")
earthquakes.sample(5)
earthquakes.iloc[0]['Date']
##following in genral correct but for the data gets an error
pd.to_datetime(earthquakes['Date'], format="%m/%d/%Y")
#solution
earthquakes['Date'] = pd.to_datetime(earthquakes['Date'], infer_datetime_format=True)
day_of_month = earthquakes['Date'].dt.day

sns.distplot(day_of_month, kde=False, bins = 31)
day_of_month.dropna(inplace=True)
sns.distplot(day_of_month, kde=False, bins = 31)


import chardet

before = "This is the euro symbol: €"
type(before)
after = before.encode("utf-8", errors='replace')
print(after)
print(after.decode('utf-8'))
after = before.encode("ascii", errors='replace')
print(after)
print(after.decode('ascii'))

# To detect the encoding:
with open("input/ks-projects-201801.csv", 'rb') as rawdata:
    result = chardet.detect(rawdata.read(40000))
with open("input/ks-projects-201801.csv", 'rb') as rawdata:
    result = chardet.detect(rawdata.read(50000))
print(result)
ks_data = pd.read_csv("input/ks-projects-201801.csv", 
encoding="utf-8")
ks_data.head()


####
import urllib
from chardet.universaldetector import  UniversalDetector
usock = open("input/PoliceKillingsUS.csv", mode='rb')
detector = UniversalDetector()
for line in usock.readlines():
    detector.feed(line)
    if detector.done:
        break;

detector.close()
usock.close()
print(detector.result)

with open("input/PoliceKillingsUS.csv", mode='rb') as rawdata:
    result = chardet.detect(rawdata.read(10000))
print(result)

with open("input/PoliceKillingsUS.csv", mode='rb') as rawdata:
    result = chardet.detect(rawdata.read(40000))
print(result)

police_killings = pd.read_csv("input/PoliceKillingsUS.csv", encoding='Windows-1252')
police_killings.head()

#save your files in utf-8 encoding
ks_data.to_csv("output/ks_utf8.csv", encoding='utf-8')
police_killings.to_csv("output/pk_utf8.csv", encoding='utf-8')


usock = open("input/character-encoding-examples/harpers_ASCII.txt", mode='rb')
detector = UniversalDetector()
for line in usock.readlines():
    detector.feed(line)
    if detector.done:
        break;

detector.close()
usock.close()
inferred_encoding = detector.result
print(inferred_encoding)


usock = open("input/character-encoding-examples/portugal_ISO-8859-1.txt", mode='rb')
detector = UniversalDetector()
for line in usock.readlines():
    detector.feed(line)
    if detector.done:
        break;

detector.close()
usock.close()
inferred_encoding = detector.result
print(inferred_encoding)


##close but needs work
file_addr = "input/character-encoding-examples/portugal_ISO-8859-1.txt"
# command = "python file_encoding_detector --file_addr {}".format(file_addr)
# enc = os.popen(command).readlines()
# enc = os.system(command)
from  file_endoding_module import figure_encoding
enc = figure_encoding(file_addr)


##############
#normalization and scaling
#if want to be comparable scaling,
# if weird looking and require normlly distributed normalization
from mlxtend.preprocessing import minmax_scaling
ks_data.dtypes
ks_data.isnull().sum()
sns.distplot(ks_data['pledged'], bins=200)
original_data = ks_data.loc[ks_data['pledged'] > 0.0, 'pledged']
normed_pledge = scipy.stats.boxcox(original_data)
sns.distplot(normed_pledge[0])

original_data = ks_data.loc[ks_data['backers'] > 0.0, 'backers']
scaled_pledge = minmax_scaling(original_data,columns = 0)
sns.distplot(scaled_pledge)

######
#inconsistent data entry
import fuzzywuzzy
from fuzzywuzzy import process


x = ["arash", "Arash", "arkjhasy", "arhasi", "arah", "arhas ", "arasw", "arash", "Arash "]
x *=10
X = pd.DataFrame({"name":x})
X.shape
X['name'] = X['name'].str.lower()
X['name'] = X['name'].str.strip()
names = X['name'].unique()
# names = names.sort()
matches = fuzzywuzzy.process.extract("arash", names, limit=5, scorer=fuzzywuzzy.fuzz.token_sort_ratio)
matches