#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 16:52:15 2021

@author: drogotti
"""

from numpy.random import seed
from numpy.random import randn
from scipy.stats import shapiro
from matplotlib import pyplot
import scipy.stats as stats
from scipy.stats import ttest_ind
from scipy.stats import spearmanr, pearsonr
import scipy.stats as stats
import pandas as pd
import numpy as np
from scipy.stats import shapiro
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew, bartlett

coviddeaths = pd.read_csv('/Users/drogotti/Downloads/AH_Cumulative_Provisional_COVID-19_Deaths_by_Sex__Race__and_Age_from_1_1_2020_to_7_28_2020.csv')
coviddeaths_small = coviddeaths.sample(100)
list(coviddeaths_small)

# Identifying the variables #

#1 factor Age has 85 levels
coviddeaths.Age Group.value_counts()
coviddeaths['Age group'].value_counts()
len(coviddeaths.Age Group.value_counts)

#1 factor Race has 6 levels
coviddeaths.Race/Ethnicity.valuecounts()
coviddeaths['Race/Ethnicity'].value_counts()
len(coviddeaths.Race/Ethnicity.valuecounts())

#1 factor MMWRWeek has 30 levels
coviddeaths.MMWRWeek.value_counts()
coviddeaths['MMWRWeek'].value_counts()
len(coviddeaths.MMWRWeek.value_counts())

#1 factor Sex has 2 levels: M or F
coviddeaths.Sex.value_counts()
coviddeaths['Sex'].value_counts()
coviddeaths['Sex'] = coviddeaths['Sex'].replace('M','0')
coviddeaths['Sex'] = coviddeaths['Sex'].replace('F','1')

coviddeaths['Sex'] = coviddeaths['Sex'].replace('0','M')
coviddeaths['Sex'] = coviddeaths['Sex'].replace('1','F')

#1 factor All Cause has 264759 levels
coviddeaths.AllCause.value_counts()
coviddeaths['AllCause'].value_counts()
len(coviddeaths.AllCause.valuecounts())

#1 factor Natural Cause has 257929 levels

coviddeaths.NaturalCause.value_counts()
coviddeaths['NaturalCause'].value_counts()
len(coviddeaths.NaturalCause.value_counts())

#1 factor Septicemia has 2470 levels
coviddeaths['[Septicemia (A40-A41)]'].value_counts()
level = coviddeaths[coviddeaths['NaturalCause'] + [coviddeaths['NaturalCause']=='level']
                    
#Combine both natural cause and natural cause - nem
coviddeaths['NaturalCause'] = coviddeaths['NaturalCause'].astype(str) + '_' + coviddeaths['NaturalCause']

#Question 1

# one continuous dependent variable- Deaths

# three independent variables age, race, MMRweek


#Question 2 

# Is there a difference between the levels of race and deaths?
# DV ~ C(IV) + C(IV)


# Create a race chart

coviddeathsrace1 = coviddeaths[coviddeaths['Race/Ethnicity']=='Hispanic']
coviddeathsrace2 = coviddeaths[coviddeaths['Race/Ethnicity']=='Non-Hispanic American Indian or Alaska Native']
coviddeathsrace3 = coviddeaths[coviddeaths['Race/Ethnicity']=='Non-Hispanic Asian']
coviddeathsrace4 = coviddeaths[coviddeaths['Race/Ethnicity']=='Non-Hispanic Black']
coviddeathsrace5 = coviddeaths[coviddeaths['Race/Ethnicity']=='Non-Hispanic White']
coviddeathsrace6 = coviddeaths[coviddeaths['Race/Ethnicity']=='Other']


# Homogeneity of Variance 
# Barlett test - Race and Age Level

stats.bartlett(coviddeathsrace1['NaturalCause'],
               coviddeathsrace2['NaturalCause'],
               coviddeathsrace3['NaturalCause'],
               coviddeathsrace4['NaturalCause'],
               coviddeathsrace5['NaturalCause'],
               coviddeathsrace6['NaturalCause'])

#P Value is greater than 0


   
# Barlett test - Race and AllCause
stats.bartlett(coviddeathsrace1['AllCause'],
               coviddeathsrace2['AllCause'],
               coviddeathsrace3['AllCause'],
               coviddeathsrace4['AllCause'],
               coviddeathsrace5['AllCause'],
               coviddeathsrace6['AllCause'])

#P Value is greater than 0

# Check for kurtosis 

print(kurtosis(coviddeathsrace1['NaturalCause']))
print(kurtosis(coviddeathsrace2['NaturalCause']))
print(kurtosis(coviddeathsrace3['NaturalCause']))
print(kurtosis(coviddeathsrace4['NaturalCause']))
print(kurtosis(coviddeathsrace5['NaturalCause']))
print(kurtosis(coviddeathsrace6['NaturalCause']))

# Leptokurtic

# Check for skewness


print(skew((coviddeathsrace1['NaturalCause'])))
print(skew((coviddeathsrace2['NaturalCause'])))
print(skew((coviddeathsrace3['NaturalCause'])))
print(skew((coviddeathsrace4['NaturalCause'])))
print(skew((coviddeathsrace5['NaturalCause'])))
print(skew((coviddeathsrace5['NaturalCause'])))

# Positive Skewed

# Histogram showing death levels in Hispanic
plt.hist(coviddeathsrace1['AllCause'])
plt.show()


# Histogram showing death levels in Non-Hispanic American Indian or Alaska Native
plt.hist(coviddeathsrace2['AllCause'])
plt.show()



plt.hist(coviddeaths['MMWRWeek'])
plt.show()

coviddeaths['MMWRWeek'].value_counts()


               
# Question 3 
# 3 1 way Anova's

# Is there a relation between Race and Natural Cause?


import statsmodels.stats.multicomp as mc
comp = mc.MultiComparison(coviddeaths['NaturalCause'], coviddeaths['Race/Ethnicity'])
post_hoc_res = comp.tukeyhsd()
tukey1way_NaturalCause_RaceEthnicity = pd.DataFrame(post_hoc_res.summary())

# Is there a relationship between the Race and All Cause?

comp = mc.MultiComparison(coviddeaths['AllCause'], coviddeaths['Race/Ethnicity'])
post_hoc_res = comp.tukeyhsd()
tukey1way_AllCause_RaceEthnicity = pd.DataFrame(post_hoc_res.summary())

# Is there a relationship between the Natural Cause and Sex?

comp = mc.MultiComparison(coviddeaths['NaturalCause'], coviddeaths['Sex'])
post_hoc_res = comp.tukeyhsd()
tukey1way_NaturalCause_Sex = pd.DataFrame(post_hoc_res.summary())
