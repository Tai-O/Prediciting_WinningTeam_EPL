
# coding: utf-8

# # Exploratory Data Analysis

# In[1]:


# Import all the necessary libraries

import math
from datetime import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import itertools
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Using pandas to load csv file into dataframe

data1 = pd.read_csv('EPL_0809.csv')
data2 = pd.read_csv('EPL_0910.csv')
data3 = pd.read_csv('EPL_1011.csv')
data4 = pd.read_csv('EPL_1112.csv')
data5 = pd.read_csv('EPL_1213.csv')
data6 = pd.read_csv('EPL_1314.csv')
data7 = pd.read_csv('EPL_1415.csv')
data8 = pd.read_csv('EPL_1516.csv')
data9 = pd.read_csv('EPL_1617.csv')
data10 = pd.read_csv('EPL_1718.csv')


# In[3]:


full_data = pd.concat([data1, data2, data3, data4, data5, data6, data7, data8, data9, data10])


# In[4]:


full_data.shape


# In[5]:


full_data.head()


# In[6]:


full_data.shape


# In[7]:


#Gets all relevant features
                      
rel_feature = ['HomeTeam','AwayTeam','FTHG','FTAG','FTR', 'HTR','HS','AS',
               'HST','AST','HF','AF','HC','AC','HY','AY','HR','AR']

new_data1 = data1[rel_feature]
new_data2 = data2[rel_feature]
new_data3 = data3[rel_feature]
new_data4 = data4[rel_feature]
new_data5 = data5[rel_feature]
new_data6 = data6[rel_feature]
new_data7 = data7[rel_feature]
new_data8 = data8[rel_feature]
new_data9 = data9[rel_feature]
new_data10 = data10[rel_feature]


# In[8]:


new_data1.tail()


# In[9]:


# Concatonating playing statistics, combines data

comb_data = pd.concat([new_data1, new_data2, new_data3, new_data4, new_data5, 
                       new_data6, new_data7, new_data8, new_data9, new_data10])


# In[10]:


comb_data.shape


# # How many matches won by the HOME teams
# # How many matches won by the AWAY teams
# # How many matches drawn

# In[11]:


def FT_data_result(new_data, year):
    return pd.DataFrame(data = [len(new_data[new_data.FTR == 'H']),
                                len(new_data[new_data.FTR == 'A']),
                                len(new_data[new_data.FTR == 'D'])],
                        index = ['Home Wins','Away Wins','Draws'],
                        columns = [year]
                        ).T

data_FTresult_agg = FT_data_result(comb_data, 'Overall')
data_FTresult1= FT_data_result(new_data1, '2008-09')
data_FTresult2= FT_data_result(new_data2, '2009-10')
data_FTresult3= FT_data_result(new_data3, '2010-11')
data_FTresult4= FT_data_result(new_data4, '2011-12')
data_FTresult5= FT_data_result(new_data5, '2012-13')
data_FTresult6= FT_data_result(new_data6, '2013-14')
data_FTresult7= FT_data_result(new_data7, '2014-15')
data_FTresult8= FT_data_result(new_data8, '2015-16')
data_FTresult9= FT_data_result(new_data9, '2016-17')
data_FTresult10 = FT_data_result(new_data10, '2017-18')


# In[12]:


#Concatonating data_result

data_FTresult = pd.concat([data_FTresult1, data_FTresult2, data_FTresult3, data_FTresult4, data_FTresult5, 
                data_FTresult6, data_FTresult7, data_FTresult8, data_FTresult9, data_FTresult10])


# In[13]:


# Plotting the result dataframe
sns.set(style="darkgrid")

FTresult_plot = data_FTresult.plot(kind = 'bar', 
                                color = ['green', 'yellow', 'red'], 
                                figsize = [10,  5], title = 'FT Result from 2008-2018')
plt.xticks(rotation = 0)
FTresult_plot.set_ylabel('Frequence', size = 12)
FTresult_plot.set_xlabel('Season', size = 12)


# In[14]:


#Plotting agg result dataframe
data_FTresult


# In[15]:


data_FTresult_agg


# In[16]:


transposed_FTresult = data_FTresult_agg.T
transposed_FTresult


# In[17]:


#Plotting FT agg result dataframe
ax1 = transposed_FTresult.plot(kind='bar', # Plot a bar chart 
                             figsize = [8, 4],  # Set size of plot in inches
                             title='Aggregated FTR Result', # Graph title
                             legend = False, # Turn the Legend off
                             width=0.5, # Set bar width as 50% of space available
                             color=[plt.cm.Paired(np.arange(len(transposed_FTresult)))]) # This changes colour of bars
                            

plt.xticks(rotation=0)
ax1.set_ylabel('Frequency', size=12)
ax1.set_xlabel('Season', size=12)


# In[18]:


# Plotting HT agg result dataframe
def HT_data_result(new_data, year):
    return pd.DataFrame(data = [len(new_data[new_data.HTR == 'H']),
                                len(new_data[new_data.HTR == 'A']),
                                len(new_data[new_data.HTR == 'D'])],
                        index = ['Home Wins','Away Wins','Draws'],
                        columns = [year]
                        ).T

data_HTresult_agg = HT_data_result(comb_data, 'Overall')
data_HTresult1= HT_data_result(new_data1, '2008-09')
data_HTresult2= HT_data_result(new_data2, '2009-10')
data_HTresult3= HT_data_result(new_data3, '2010-11')
data_HTresult4= HT_data_result(new_data4, '2011-12')
data_HTresult5= HT_data_result(new_data5, '2012-13')
data_HTresult6= HT_data_result(new_data6, '2013-14')
data_HTresult7= HT_data_result(new_data7, '2014-15')
data_HTresult8= HT_data_result(new_data8, '2015-16')
data_HTresult9= HT_data_result(new_data9, '2016-17')
data_HTresult10 = HT_data_result(new_data10, '2017-18')


# In[19]:


transposed_HTresult = data_HTresult_agg.T
transposed_HTresult


# In[20]:


# Checking results at HT
ax1 = transposed_HTresult.plot(kind='bar', # Plot a bar chart 
                             figsize = [8,4],  # Set size of plot in inches
                             title='Aggregated HT Result', # Graph title
                             legend = False, # Turn the Legend off
                             width=0.5, # Set bar width as 50% of space available
                             color=[plt.cm.Paired(np.arange(len(transposed_HTresult)))]) # This changes colour of bars
                            

plt.xticks(rotation=0)
ax1.set_ylabel('Frequency', size=12)
ax1.set_xlabel('Season', size=12)


# In[21]:


#dataframe with two features (FTR, HTR)
                      
n_data1 = data1[rel_feature[4:6]]
n_data2 = data2[rel_feature[4:6]]
n_data3 = data3[rel_feature[4:6]]
n_data4 = data4[rel_feature[4:6]]
n_data5 = data5[rel_feature[4:6]]
n_data6 = data6[rel_feature[4:6]]
n_data7 = data7[rel_feature[4:6]]
n_data8 = data8[rel_feature[4:6]]
n_data9 = data9[rel_feature[4:6]]
n_data10 = data10[rel_feature[4:6]]


# In[22]:


combined_data = pd.concat([n_data1, n_data2, n_data3, n_data4, n_data5, n_data6, n_data7, n_data8, n_data9, n_data10])


# In[23]:


combined_data.tail()


# In[24]:


#Check relationships between HTR and FTR

combined_data['Outcome'] = np.where(combined_data['FTR'] == combined_data['HTR'], 'no change', 'changed')


# In[25]:


combined_data.tail()


# In[26]:


sns.set(style="darkgrid")
ax = sns.countplot(y="Outcome", data=combined_data)


# # FTR Percentage

# In[27]:


data_result = data_FTresult_agg.T


# In[28]:


(100. * data_result / data_result.sum()).round(0)
plot = data_result.plot.pie(y = 'Overall', figsize = (7, 8), autopct='%1.1f%%', startangle=20, fontsize= 15)


# In[29]:


comb_data.head()


# In[30]:


final_data = comb_data.to_csv('final_data.csv', header = True, index = False)


# In[31]:


#convert 'FTR' and 'HTR' to numeric

#comb_data['FTR'].replace(['H', 'D', 'A'], [2, 1, 0], inplace = True)
#comb_data['HTR'].replace(['H', 'D', 'A'], [2, 1, 0], inplace = True)


# In[32]:


comb_data.tail()


# In[55]:


import seaborn as sns
#Correlation Matrix
corr = comb_data.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
plt.subplots(figsize=(16,12))
with sns.axes_style("white"):
    ax = sns.heatmap(corr, mask=mask, vmax=.3, square=True)
    plt.title('Heatmap of Correlation Matrix')


# In[37]:


#sns.pairplot(dummy_data, x_vars = ['HTR_H', 'HTR_D', 'HTR_A'], y_vars = 'FTR_H', size = 7, aspect =0.7)


# In[38]:


data1.keys()

