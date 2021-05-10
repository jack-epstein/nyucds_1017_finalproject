#!/usr/bin/env python
# coding: utf-8

"""This file prints cleaner images from the EDA notebook"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

data_dir = '/Users/jackepstein/Documents/GitHub/heartdisease-1017/kaggle'

def getDfSummary(input_data):
    "Gives a slightly more robust initial EDA compared to pd.decribe()"
    #start by getting the stats that describe already gives us, 
        #then transpose this (gets us mean, max, min, std, 25%, 50%, 75%)
    output_data = input_data.describe()
    output_data = output_data.transpose()
    
    #get the median of each column as well
    output_data['median'] = np.median(input_data, axis=0)
    
    #to get distinct counts, first use the nunique function
    #turn this series into a new data frame and merge it with the one from above
    uniques = input_data.nunique(0)
    ph = uniques.to_frame(name='number_distinct')
    output_data = pd.merge(output_data,ph,left_index=True,right_index=True)
        
    #using the count total, get the total number of rows and take this difference to get number_nan
    numrows = len(input_data.index)
    output_data['number_nan'] = output_data['count'].apply(lambda x: numrows - x) 

    return output_data


df = pd.read_csv(os.path.join(data_dir, 'heart.csv'))
getDfSummary(df)


#split out columns by continuous features, discrete features and the target
cont_feats = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
discrete_feats = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
target = ['target']


#do pairplots only on continuous features - with target overlay
sns.pairplot(df[cont_feats+target], hue='target')
plt.show()


#get separate dataframes for each class of target
df_pos = df.loc[df.target==1].astype(object)
df_neg = df.loc[df.target==0].astype(object)


col = ['sex', 'cp']

fig, ax = plt.subplots(1,2, figsize=(16,6))
fig.suptitle('Select Discrete Features by Target')

for i in range(2):
    x = np.arange(len(df_neg[col[i]].value_counts())) #number of ticks
    y1 = df_neg[col[i]].value_counts().sort_index() #targ=0 datapoints
    y2 = df_pos[col[i]].value_counts().sort_index() #targ=1 datapoints 
    width = 0.3
  
    # plot data in grouped manner of bar type
    ax[i].bar(x-0.15, y1, width, label='Target=0')
    ax[i].bar(x+0.15, y2, width, label='Target=1')
    
    ax[i].set_ylabel('Number of Observations')
    ax[i].legend(loc='best')
    ax[i].set_xticks(x)
    

ax[0].set_xticklabels(('Women', 'Men'))
ax[1].set_xticklabels(('Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'))
ax[0].set_title('Sex')
ax[1].set_title('Chest Pain')

plt.show()


#use the sns plot for age and oldpeak dist

fig, ax = plt.subplots(1,2, figsize=(16,6))    

# Draw the density plots
sns.distplot(df_neg.age, hist = False, kde = True, ax=ax[0],
             kde_kws = {'shade': True},
             label = 'Target=0')
sns.distplot(df_pos.age, hist = False, kde = True, ax=ax[0],
             kde_kws = {'shade': True},
             label = 'Target=1')

sns.distplot(df_neg.oldpeak, hist = False, kde = True, ax=ax[1],
             kde_kws = {'shade': True},
             label = 'Target=0')
sns.distplot(df_pos.oldpeak, hist = False, kde = True, ax=ax[1],
             kde_kws = {'shade': True},
             label = 'Target=1')

# Plot formatting
ax[0].set_title('Age Distribution by Target')
ax[1].set_title('Oldpeak Distrubution by Target')
for i in range(2):
    ax[i].legend(loc='best')
    ax[i].set_ylabel('Density')
    
plt.show()


#create dfs for subplots
df_w = df.loc[df.sex==1].astype(object)
df_m = df.loc[df.sex==0].astype(object)


#create dfs for subplots
df_cp0 = df.loc[df.cp==0].astype(object)
df_cp1 = df.loc[df.cp==1].astype(object)
df_cp2 = df.loc[df.cp==2].astype(object)
df_cp3 = df.loc[df.cp==3].astype(object)


fig, ax = plt.subplots(1,2, figsize=(16,6))

x = np.arange(2) 
y1 = df_w.target.value_counts().sort_index() 
y2 = df_m.target.value_counts().sort_index() 
y3 = df_cp0.target.value_counts().sort_index() 
y4 = df_cp1.target.value_counts().sort_index() 
y5 = df_cp2.target.value_counts().sort_index() 
y6 = df_cp3.target.value_counts().sort_index() 

width_0 = 0.3
width_1 = 0.15


# plot data in grouped manner of bar type
ax[0].bar(x-0.15, y1, width_0, label='Women')
ax[0].bar(x+0.15, y2, width_0, label='Men')

ax[1].bar(x-0.225, y3, width_1, label='cp=0', color='tab:blue')
ax[1].bar(x-0.075, y4, width_1, label='cp=1', color='tab:orange')
ax[1].bar(x+0.075, y5, width_1, label='cp=2', color='tab:purple')
ax[1].bar(x+0.225, y6, width_1, label='cp=3', color='tab:olive')

ax[0].set_title('Target Distribution by Sex')
ax[1].set_title('Target Distribution by Chest Pain')

for i in range(2):
    ax[i].set_ylabel('Number of Observations')
    ax[i].legend(loc='best')

    ax[i].set_xticks(x)
    ax[i].set_xticklabels(('Neg', 'Pos'))

plt.show()


# ### Correlation Matrix


#full correlation matrix
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), cmap='Blues')
plt.show()

df.corr()