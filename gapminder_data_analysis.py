# -*- coding: utf-8 -*-
"""
Created on Tue Aug 03 15:33:33 2020

@author: Jonathan Ting
"""

# Import required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi
import scipy.stats

# Bug fix for display formats and change settings to show all rows and columns
pd.set_option('display.float_format', lambda x:'%f'%x)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Read in the GapMinder dataset
raw_data = pd.read_csv('./gapminder.csv', low_memory=False)

# Report facts regarding the original dataset
print("Facts regarding the original GapMinder dataset:")
print("---------------------------------------")
print("Number of countries: {0}".format(len(raw_data)))
print("Number of variables: {0}\n".format(len(raw_data.columns)))
print("All variables:\n{0}\n".format(list(raw_data.columns)))
print("Data types of each variable:\n{0}\n".format(raw_data.dtypes))
print("First 5 rows of entries:\n{0}\n".format(raw_data.head()))
print("=====================================\n")

# Choose variables of interest
# var_of_int = ['country', 'incomeperperson', 'alcconsumption', 'co2emissions', 
# 'internetuserate', 'oilperperson', 'relectricperperson', 'urbanrate']
var_of_int = ['oilperperson', 'relectricperperson', 'urbanrate']
print("Chosen variables of interest:\n{0}\n".format(var_of_int))
print("=====================================\n")

# Code out missing values by replacing with NumPy's NaN data type
data = raw_data[var_of_int].replace(' ', np.nan)
print("Replaced missing values with NaNs:\n{0}\n".format(data.head()))
print("=====================================\n")

# Cast the numeric variables to the appropriate data type then quartile split
numeric_vars = var_of_int[:]
for var in numeric_vars: data[var] = pd.to_numeric(data[var], downcast='float', errors='raise')
print("Simple statistics of each variable:\n{0}\n".format(data.describe()))
print("=====================================\n")

# Create secondary variables to investigate frequency distributions
print("Separate continuous values categorically using secondary variables:")
print("---------------------------------------")
data['oilpp (tonnes)'] = pd.cut(data['oilperperson'], 4)
oil_val_count = data.groupby('oilpp (tonnes)').size()
oil_dist = data['oilpp (tonnes)'].value_counts(sort=False, dropna=True, normalize=True)
oil_freq_tab = pd.concat([oil_val_count, oil_dist], axis=1)
oil_freq_tab.columns = ['value_count', 'frequency']
print("Frequency table of oil consumption per person:\n{0}\n".format(oil_freq_tab))

data['relectricpp (kWh)'] = pd.cut(data['relectricperperson'], 2)
elec_val_count = data.groupby('relectricpp (kWh)').size()
elec_dist = data['relectricpp (kWh)'].value_counts(sort=False, dropna=True, normalize=True)
elec_freq_tab = pd.concat([elec_val_count, elec_dist], axis=1)
elec_freq_tab.columns = ['value_count', 'frequency']
print("Frequency table of residential electricity consumption per person:\n{0}\n".format(elec_freq_tab))

data['urbanr (%)'] = pd.cut(data['urbanrate'], 4)
urb_val_count = data.groupby('urbanr (%)').size()
urb_dist = data['urbanr (%)'].value_counts(sort=False, dropna=True, normalize=True)
urb_freq_tab = pd.concat([urb_val_count, urb_dist], axis=1)
urb_freq_tab.columns = ['value_count', 'frequency']
print("Frequency table of urban population:\n{0}\n".format(urb_freq_tab))
print("=====================================\n")

# Code in valid data in place of missing data for each variable
print("Number of missing data in variables:")
print("oilperperson: {0}".format(data['oilperperson'].isnull().sum()))
print("relectricperperson: {0}".format(data['relectricperperson'].isnull().sum()))
print("urbanrate: {0}\n".format(data['urbanrate'].isnull().sum()))
print("=====================================\n")

print("Investigate entries with missing urbanrate data:\n{0}\n".format(data[['oilperperson', 'relectricperperson']][data['urbanrate'].isnull()]))
print("Data for other variables are also missing for 90% of these entries.")
print("Therefore, eliminate them from the dataset.\n")
data = data[data['urbanrate'].notnull()]
print("=====================================\n")

null_elec_data = data[data['relectricperperson'].isnull()].copy()
print("Investigate entries with missing relectricperperson data:\n{0}\n".format(null_elec_data.head()))
elec_map_dict = data.groupby('urbanr (%)').median()['relectricperperson'].to_dict()
print("Median values of relectricperperson corresponding to each urbanrate group:\n{0}\n".format(elec_map_dict))
null_elec_data['relectricperperson'] = null_elec_data['urbanr (%)'].map(elec_map_dict)
data = data.combine_first(null_elec_data)
data['relectricpp (kWh)'] = pd.cut(data['relectricperperson'], 2)
print("Replace relectricperperson NaNs based on their quartile group's median:\n{0}\n".format(data.head()))
print("-------------------------------------\n")

null_oil_data = data[data['oilperperson'].isnull()].copy()
oil_map_dict = data.groupby('urbanr (%)').median()['oilperperson'].to_dict()
print("Median values of oilperperson corresponding to each urbanrate group:\n{0}\n".format(oil_map_dict))
null_oil_data['oilperperson'] = null_oil_data['urbanr (%)'].map(oil_map_dict)
data = data.combine_first(null_oil_data)
data['oilpp (tonnes)'] = pd.cut(data['oilperperson'], 4)
print("Replace oilperperson NaNs based on their quartile group's median:\n{0}\n".format(data.head()))
print("=====================================\n")

# Investigate the new frequency distributions
print("Report the new frequency table for each variable:")
print("---------------------------------------")
oil_val_count = data.groupby('oilpp (tonnes)').size()
oil_dist = data['oilpp (tonnes)'].value_counts(sort=False, dropna=True, normalize=True)
oil_freq_tab = pd.concat([oil_val_count, oil_dist], axis=1)
oil_freq_tab.columns = ['value_count', 'frequency']
print("Frequency table of oil consumption per person:\n{0}\n".format(oil_freq_tab))

elec_val_count = data.groupby('relectricpp (kWh)').size()
elec_dist = data['relectricpp (kWh)'].value_counts(sort=False, dropna=True, normalize=True)
elec_freq_tab = pd.concat([elec_val_count, elec_dist], axis=1)
elec_freq_tab.columns = ['value_count', 'frequency']
print("Frequency table of residential electricity consumption per person:\n{0}\n".format(elec_freq_tab))

urb_val_count = data.groupby('urbanr (%)').size()
urb_dist = data['urbanr (%)'].value_counts(sort=False, dropna=True, normalize=True)
urb_freq_tab = pd.concat([urb_val_count, urb_dist], axis=1)
urb_freq_tab.columns = ['value_count', 'frequency']
print("Frequency table of urban population:\n{0}\n".format(urb_freq_tab))
print("=====================================\n")

#==============================================================================
# # Visualize the data
# mpl.rcParams['axes.titlesize'] = 15
# mpl.rcParams['axes.labelsize'] = 12
# print("Univariate graphs:")
# print("-------------------------------------")
# sns.distplot(data['oilperperson'], kde=False)
# plt.xlabel("Oil consumption per person (tonnes)")
# plt.ylabel("Number of countries")
# plt.title("Distribution of oil consumption per person")
# plt.show()
# sns.distplot(data['relectricperperson'], kde=False)
# plt.xlabel("Residential electricity consumption per person (kWh)")
# plt.ylabel("Number of countries")
# plt.title("Distribution of residential electricity consumption per person")
# plt.show()
# sns.distplot(data['urbanrate'], kde=False)
# plt.xlabel("Urban population (%)")
# plt.ylabel("Number of countries")
# plt.title("Distribution of urban population")
# plt.show()
# 
# print("\nBivariate graphs:")
# print("-------------------------------------")
# sns.regplot(x='oilperperson', y='relectricperperson', data=data)
# plt.xlabel("Oil consumption per person (tonnes)")
# plt.ylabel("Residential electricity consumption per person (kWh)")
# plt.title("Scatter plot of oil vs residential electricity consumption per person")
# plt.show()
# sns.regplot(x='urbanrate', y='relectricperperson', data=data)
# plt.xlabel("Urban population (%)")
# plt.ylabel("Residential electricity consumption per person (kWh)")
# plt.title("Scatter plot of urban population vs residential electricity consumption per person")
# plt.show()
# print("=====================================\n")

# # Run ANOVA
# print("ANOVA results:")
# print("-------------------------------------")
# model1 = smf.ols(formula='relectricperperson ~ C(Q("oilpp (tonnes)"))', data=data)
# m1, std1 = data[['relectricperperson', 'oilpp (tonnes)']].groupby('oilpp (tonnes)').mean(), data[['relectricperperson', 'oilpp (tonnes)']].groupby('oilpp (tonnes)').std()
# mc1 = multi.MultiComparison(data['relectricperperson'], data['oilpp (tonnes)'])
# print("relectricperperson ~ oilpp (tonnes)\n{0}".format(model1.fit().summary()))
# print("Means for relectricperperson by oilpp status:\n{0}\n".format(m1))
# print("Standard deviations for relectricperperson by oilpp status:\n{0}\n".format(std1))
# print("MultiComparison summary:\n{0}\n".format(mc1.tukeyhsd().summary()))
# 
# model2 = smf.ols(formula='relectricperperson ~ C(Q("urbanr (%)"))', data=data)
# m2, std2 = data[['relectricperperson', 'urbanr (%)']].groupby('urbanr (%)').mean(), data[['relectricperperson', 'urbanr (%)']].groupby('urbanr (%)').std()
# mc2 = multi.MultiComparison(data['relectricperperson'], data['urbanr (%)'])
# print("relectricperperson ~ urbanr (%)\n{0}".format(model2.fit().summary()))
# print("Means for relectricperperson by urbanr status:\n{0}\n".format(m2))
# print("Standard deviations for relectricperperson by urbanr status:\n{0}\n".format(std2))
# print("MultiComparison summary:\n{0}\n".format(mc2.tukeyhsd().summary()))

# # Run Chi-Square Test for Independence
# print("Chi-square test for independence results:")
# print("-------------------------------------")
# ct1 = pd.crosstab(data[['relectricpp (kWh)', 'oilpp (tonnes)']]['relectricpp (kWh)'], data[['relectricpp (kWh)', 'oilpp (tonnes)']]['oilpp (tonnes)'])
# colpct1 = ct1 / ct1.sum(axis=0)
# cs1 = scipy.stats.chi2_contingency(ct1)
# print("Contingency table of observed counts:\n{0}\n".format(ct1))
# print("Column percentages:\n{0}\n".format(colpct1))
# print('chi-square value, p value, expected counts:\n{0}\n'.format(cs1))
# 
# print("Post hoc chi-square test for independence results:")
# print("-------------------------------------")
# recode1 = {'(0.0201, 3.0814]': '(0.0201, 3.0814]', '(3.0814, 6.13]': '(3.0814, 6.13]'}
# data['COMP1v2']= data['oilpp (tonnes)'].map(recode1)
# recode2 = {'(0.0201, 3.0814]': '(0.0201, 3.0814]', '(6.13, 9.18]': '(6.13, 9.18]'}
# data['COMP1v3']= data['oilpp (tonnes)'].map(recode2)
# recode3 = {'(0.0201, 3.0814]': '(0.0201, 3.0814]', '(9.18, 12.229]': '(9.18, 12.229]'}
# data['COMP1v4']= data['oilpp (tonnes)'].map(recode3)
# recode4 = {'(3.0814, 6.13]': '(3.0814, 6.13]', '(6.13, 9.18]': '(6.13, 9.18]'}
# data['COMP2v3']= data['oilpp (tonnes)'].map(recode4)
# recode5 = {'(3.0814, 6.13]': '(3.0814, 6.13]', '(9.18, 12.229]': '(9.18, 12.229]'}
# data['COMP2v4']= data['oilpp (tonnes)'].map(recode5)
# recode6 = {'(6.13, 9.18]': '(6.13, 9.18]', '(9.18, 12.229]': '(9.18, 12.229]'}
# data['COMP3v4']= data['oilpp (tonnes)'].map(recode6)
# 
# ct2 = pd.crosstab(data[['relectricpp (kWh)', 'COMP1v2']]['relectricpp (kWh)'], data[['relectricpp (kWh)', 'COMP1v2']]['COMP1v2'])
# colpct2 = ct2 / ct2.sum(axis=0)
# cs2 = scipy.stats.chi2_contingency(ct2)
# print("Contingency table of observed counts:\n{0}\n".format(ct2))
# print("Column percentages:\n{0}\n".format(colpct2))
# print('chi-square value, p value, expected counts:\n{0}\n'.format(cs2))
# 
# ct3 = pd.crosstab(data[['relectricpp (kWh)', 'COMP1v3']]['relectricpp (kWh)'], data[['relectricpp (kWh)', 'COMP1v3']]['COMP1v3'])
# colpct3 = ct3 / ct3.sum(axis=0)
# cs3 = scipy.stats.chi2_contingency(ct3)
# print("Contingency table of observed counts:\n{0}\n".format(ct3))
# print("Column percentages:\n{0}\n".format(colpct3))
# print('chi-square value, p value, expected counts:\n{0}\n'.format(cs3))
# 
# ct4 = pd.crosstab(data[['relectricpp (kWh)', 'COMP1v4']]['relectricpp (kWh)'], data[['relectricpp (kWh)', 'COMP1v4']]['COMP1v4'])
# colpct4 = ct4 / ct4.sum(axis=0)
# cs4 = scipy.stats.chi2_contingency(ct4)
# print("Contingency table of observed counts:\n{0}\n".format(ct4))
# print("Column percentages:\n{0}\n".format(colpct4))
# print('chi-square value, p value, expected counts:\n{0}\n'.format(cs4))
# 
# ct5 = pd.crosstab(data[['relectricpp (kWh)', 'COMP2v3']]['relectricpp (kWh)'], data[['relectricpp (kWh)', 'COMP2v3']]['COMP2v3'])
# colpct5 = ct5 / ct5.sum(axis=0)
# cs5 = scipy.stats.chi2_contingency(ct5)
# print("Contingency table of observed counts:\n{0}\n".format(ct5))
# print("Column percentages:\n{0}\n".format(colpct5))
# print('chi-square value, p value, expected counts:\n{0}\n'.format(cs5))
# 
# ct6 = pd.crosstab(data[['relectricpp (kWh)', 'COMP2v4']]['relectricpp (kWh)'], data[['relectricpp (kWh)', 'COMP2v4']]['COMP2v4'])
# colpct6 = ct6 / ct6.sum(axis=0)
# cs6 = scipy.stats.chi2_contingency(ct6)
# print("Contingency table of observed counts:\n{0}\n".format(ct6))
# print("Column percentages:\n{0}\n".format(colpct6))
# print('chi-square value, p value, expected counts:\n{0}\n'.format(cs6))
# 
# ct7 = pd.crosstab(data[['relectricpp (kWh)', 'COMP3v4']]['relectricpp (kWh)'], data[['relectricpp (kWh)', 'COMP3v4']]['COMP3v4'])
# colpct7 = ct7 / ct7.sum(axis=0)
# cs7 = scipy.stats.chi2_contingency(ct7)
# print("Contingency table of observed counts:\n{0}\n".format(ct7))
# print("Column percentages:\n{0}\n".format(colpct7))
# print('chi-square value, p value, expected counts:\n{0}\n'.format(cs7))

# # Generate correlation coefficients
# print("Generate correlation coefficients:")
# print('Association between relectricperperson and oilperperson:\n{0}\n'.format(scipy.stats.pearsonr(data['relectricperperson'], data['oilperperson'])))
# print('Association between relectricperperson and urbanrate:\n{0}\n'.format(scipy.stats.pearsonr(data['relectricperperson'], data['urbanrate'])))
# 
#==============================================================================

# Testing for moderator effect
print("Testing for moderator effect:")
print("-------------------------------------")
print("Urban population as moderator:")
urb1 = data[data['urbanr (%)'] == '(10.31, 32.8]']
urb2 = data[data['urbanr (%)'] == '(32.8, 55.2]']
urb3 = data[data['urbanr (%)'] == '(55.2, 77.6]']
urb4 = data[data['urbanr (%)'] == '(77.6, 100]']
print('Association between relectricperperson and oilperperson for urbanr (%) (10.31, 32.8]:\n{0}\n'
      .format(scipy.stats.pearsonr(urb1['relectricperperson'], urb1['oilperperson'])))
print('Association between relectricperperson and oilperperson for urbanr (%) (32.8, 55.2]:\n{0}\n'
.format(scipy.stats.pearsonr(urb2['relectricperperson'], urb2['oilperperson'])))
print('Association between relectricperperson and oilperperson for urbanr (%) (55.2, 77.6]:\n{0}\n'
      .format(scipy.stats.pearsonr(urb3['relectricperperson'], urb3['oilperperson'])))
print('Association between relectricperperson and oilperperson for urbanr (%) (77.6, 100]:\n{0}\n'
      .format(scipy.stats.pearsonr(urb4['relectricperperson'], urb4['oilperperson'])))

print("\nBivariate graphs:")
print("-------------------------------------")
sns.regplot(x='oilperperson', y='relectricperperson', data=urb1)
plt.xlabel("Oil consumption per person (tonnes)")
plt.ylabel("Residential electricity consumption per person (kWh)")
plt.title("Scatter plot for urbanr (%) (10.31, 32.8]")
plt.show()
sns.regplot(x='oilperperson', y='relectricperperson', data=urb2)
plt.xlabel("Oil consumption per person (tonnes)")
plt.ylabel("Residential electricity consumption per person (kWh)")
plt.title("Scatter plot for urbanr (%) (32.8, 55.2]")
plt.show()
sns.regplot(x='oilperperson', y='relectricperperson', data=urb3)
plt.xlabel("Oil consumption per person (tonnes)")
plt.ylabel("Residential electricity consumption per person (kWh)")
plt.title("Scatter plot for urbanr (%) (55.2, 77.6]")
plt.show()
sns.regplot(x='oilperperson', y='relectricperperson', data=urb4)
plt.xlabel("Oil consumption per person (tonnes)")
plt.ylabel("Residential electricity consumption per person (kWh)")
plt.title("Scatter plot for urbanr (%) (77.6, 100]")
plt.show()
