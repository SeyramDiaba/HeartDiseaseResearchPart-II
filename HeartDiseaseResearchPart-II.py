# import libraries
import codecademylib3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns


# load data
heart = pd.read_csv('heart_disease.csv')

# Inspecting first five rows of data
print(heart.head())

# side by side boxplots of patients who were and not diagnosed of heart disease
sns.boxplot(x = heart.heart_disease, y= heart.thalach)
plt.show()
plt.clf()

# Savinig values of patients who were and not diagnosed of heart disease into seperate variables
thalach_hd = heart.thalach[heart.heart_disease == 'presence']

thalach_no_hd = heart.thalach[heart.heart_disease == 'absence']


# Difference in mean thalach for both diagnosed and non-diagnosed patients
thalach_hd_mean = np.mean(thalach_hd)
print('Thalach HD average: ',thalach_hd_mean)

thalach_no_hd_mean = np.mean(thalach_no_hd)
print('Thalach no HD average: ',thalach_no_hd_mean)

print("Thalach difference in mean:",thalach_no_hd_mean - thalach_hd_mean)
print('------------------------------------')

# Difference in median thalach for both diagnosed and non-diagnosed patients
thalach_hd_median = np.median(thalach_hd)
print("Thalach HD median:", thalach_hd_median)

thalach_no_hd_median = np.median(thalach_no_hd)
print("Thalach no HD median: ", thalach_no_hd_median)

print('Thalach median differences: ', thalach_no_hd_median - thalach_hd_median)
print('------------------------------------')

# Importing the statistical test from scipy.stats
from scipy.stats import ttest_ind
tstat, pval = ttest_ind(thalach_hd, thalach_no_hd)
print('P-Value, for "thalach" two sample ttest: ',pval) #pval is less than 0.05 so we drop the null hypothesis

# age
# Investigating for the relation between age and HD
sns.boxplot(x= heart['heart_disease'], y = heart['age'])
plt.show()
plt.clf()
print('------------------------------------')

# Mean age difference between patients diagnosed and not diagnosed with HD
age_hd = heart.age[heart['heart_disease']=='presence']
age_no_hd = heart.age[heart['heart_disease']== 'absence']
age_hd_mean= np.mean(age_hd)
print('Mean Age of HD patients: ', age_hd_mean)
age_no_hd_mean= np.mean(age_no_hd)
print('Mean Age of patients with no HD: ',age_no_hd_mean)
print('Difference in mean ages: ',age_hd_mean-age_no_hd_mean)
print('------------------------------------')

# Median age difference between patients diagnosed and not diagnosed with HD
age_hd_median = np.median(age_hd)
print('Median Age of HD patients: ',age_hd_median)
age_no_hd_median = np.median(age_no_hd)
print('Median Age for no HD patients: ',age_no_hd_median )
print('Median differences in ages: ', age_hd_median- age_no_hd_median)
print('------------------------------------')

# Calculating pval for the relationship between ages and heart disease
tstat,pval = ttest_ind(age_hd,age_no_hd)
print('P-Value of two sample ttest between ages of patients diagnosed and not diagnosed of HD :',pval)# pval is less than 0.05 so we drop the null hypothesis


# Investigating chest pain and max heart rate
sns.boxplot(x=heart.cp, y=heart.thalach)
plt.show()
plt.clf()

# Values of thalech for patients who experienced each type of chest pain.
thalach_typical = heart.thalach[heart['cp']== 'typical angina']
thalach_asymptom = heart.thalach[heart['cp']== 'asymptomatic']
thalach_nonangin = heart.thalach[heart['cp']== 'non-anginal pain']
thalach_atypical = heart.thalach[heart['cp']== 'atypical angina']
print('------------------------------------')

#Run a single hypothesis test to address the following null and alternative hypotheses: Null: People with typical angina, non-anginal pain, atypical angina, and asymptomatic people all have the same average thalach Alternative: People with typical angina, non-anginal pain, atypical angina, and asymptomatic people do not all have the same average thalach.

# We need ANOVA to work this hypothesis
from scipy.stats import f_oneway
fstat,pval = f_oneway(thalach_typical,thalach_asymptom,thalach_nonangin,thalach_atypical)
print('P-Value of cp values',pval)# pval is less than 0.05 so we drop null hypothesis, and we conclude there is at lease one pair of chest pain types that have significantly different average max heart rates during exercise(thalach)

# Determine which of the pairs are significantly different
# we need to perform a tukey test to achieve this.
from statsmodels.stats.multicomp import pairwise_tukeyhsd
tukey_results = pairwise_tukeyhsd(heart.thalach,heart.cp,0.05)
print('tukey pval',tukey_results)

# Investigating Heart Disease and Chest Pain (both variables are categorical)
Xtab = pd.crosstab(heart.heart_disease,heart.cp)
print(Xtab)

# Hypothsis test for the two categorical variables 'heart_disease' and 'cp'
from scipy.stats import chi2_contingency
chi2,pval,dof,expected = chi2_contingency(Xtab)
print('pval of two categorical variables"cp & heart disease":', pval)

# Investigating relation between sex and heart disease (two cat vars)
sex_heart_xtab = pd.crosstab(heart.sex,heart.heart_disease)
print(sex_heart_xtab)

chi2,pval,dof,expected = chi2_contingency(sex_heart_xtab)
print('P-Value of alleged relationship b/n "sex" and "heart disease"',pval) #pval is less than 0.05 meaning we drop the null hypothesis that "Both sex' are prone to developing heart disease"




