#!/usr/bin/env python
# coding: utf-8

# 

# Dataset is downloaded from here: https://www.kaggle.com/amitabhajoy/bengaluru-house-price-data

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib 
matplotlib.rcParams["figure.figsize"] = (20,10)


# In[3]:


df1 = pd.read_csv("Bengaluru_House_Data.csv")
df1.head()


# In[3]:


df1.shape


# In[4]:


df1.columns


# In[5]:


df1['area_type'].unique()


# In[6]:


df1['area_type'].value_counts()


# **Drop features that are not required to build our model**

# In[7]:


df2 = df1.drop(['area_type','society','balcony','availability'],axis='columns')
df2.shape


# In[8]:


df2.isnull().sum()


# In[9]:


df2.shape


# In[10]:


df3 = df2.dropna()
df3.isnull().sum()


# In[11]:


df3.shape


# In[12]:


df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))
df3.bhk.unique()


# **Explore total_sqft feature**

# In[13]:


def is_float(x):
    try:
        float(x)
    except:
        return False
    return True


# In[14]:


2+3


# In[15]:


df3[~df3['total_sqft'].apply(is_float)].head(10)


# In[16]:


def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None   


# In[17]:


df4 = df3.copy()
df4.total_sqft = df4.total_sqft.apply(convert_sqft_to_num)
df4 = df4[df4.total_sqft.notnull()]
df4.head(2)


# **For below row, it shows total_sqft as 2475 which is an average of the range 2100-2850**

# In[18]:


df4.loc[30]


# In[19]:


(2100+2850)/2


# In[20]:


df5 = df4.copy()
df5['price_per_sqft'] = df5['price']*100000/df5['total_sqft']
df5.head()


# In[21]:


df5_stats = df5['price_per_sqft'].describe()
df5_stats


# In[69]:


df5.to_csv("bhp.csv",index=False)


# **Examine locations which is a categorical variable. We need to apply dimensionality reduction technique here to reduce number of locations**

# In[22]:


df5.location = df5.location.apply(lambda x: x.strip())
location_stats = df5['location'].value_counts(ascending=False)
location_stats


# In[23]:


location_stats.values.sum()


# In[24]:


len(location_stats[location_stats>10])


# In[25]:


len(location_stats)


# In[26]:


len(location_stats[location_stats<=10])


# In[27]:


location_stats_less_than_10 = location_stats[location_stats<=10]
location_stats_less_than_10


# In[28]:


len(df5.location.unique())


# In[29]:


df5.location = df5.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
len(df5.location.unique())


# In[30]:


df5.head(10)


# In[31]:


df5[df5.total_sqft/df5.bhk<300].head()


# In[32]:


df5.shape


# In[33]:


df6 = df5[~(df5.total_sqft/df5.bhk<300)]
df6.shape


# In[34]:


df6.price_per_sqft.describe()


# **Here we find that min price per sqft is 267 rs/sqft whereas max is 12000000, this shows a wide variation in property prices. We should remove outliers per location using mean and one standard deviation**

# In[35]:


def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
df7 = remove_pps_outliers(df6)
df7.shape


# **Let's check if for a given location how does the 2 BHK and 3 BHK property prices look like**

# In[38]:


def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
df8 = remove_bhk_outliers(df7)
# df8 = df7.copy()
df8.shape


# **Plot same scatter chart again to visualize price_per_sqft for 2 BHK and 3 BHK properties**

# In[41]:


import matplotlib
matplotlib.rcParams["figure.figsize"] = (20,10)
plt.hist(df8.price_per_sqft,rwidth=0.8)
plt.xlabel("Price Per Square Feet")
plt.ylabel("Count")


# In[42]:


df8.bath.unique()


# In[44]:


df8[df8.bath>10]


# **It is unusual to have 2 more bathrooms than number of bedrooms in a home**

# In[45]:


df8[df8.bath>df8.bhk+2]


# In[46]:


df9 = df8[df8.bath<df8.bhk+2]
df9.shape


# In[47]:


df9.head(2)


# In[48]:


df10 = df9.drop(['size','price_per_sqft'],axis='columns')
df10.head(3)


# In[49]:


dummies = pd.get_dummies(df10.location)
dummies.head(3)


# In[50]:


df11 = pd.concat([df10,dummies.drop('other',axis='columns')],axis='columns')
df11.head()


# In[51]:


df12 = df11.drop('location',axis='columns')
df12.head(2)


# <h2 style='color:blue'>Build a Model Now...</h2>

# In[52]:


df12.shape


# In[53]:


X = df12.drop(['price'],axis='columns')
X.head(3)


# In[54]:


X.shape


# In[55]:


y = df12.price
y.head(3)


# In[56]:


len(y)


# In[57]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)


# In[58]:


from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(X_train,y_train)
lr_clf.score(X_test,y_test)


# In[61]:


def predict_price(location,sqft,bath,bhk):    
    loc_index = np.where(X.columns==location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return lr_clf.predict([x])[0]


# In[62]:


predict_price('1st Phase JP Nagar',1000, 2, 2)


# In[63]:


predict_price('1st Phase JP Nagar',1000, 3, 3)


# In[64]:


predict_price('Indira Nagar',1000, 2, 2)


# In[65]:


predict_price('Indira Nagar',1000, 3, 3)

