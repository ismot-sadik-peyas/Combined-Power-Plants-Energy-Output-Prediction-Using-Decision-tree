#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics


# In[26]:


data = pd.read_csv(r"C:\Users\peash\Desktop\cse445 project\dataset.csv")
df = pd.DataFrame(data)
df.info()
data.head()


# In[27]:


df.describe()


# In[28]:


plt.figure(figsize = (7, 6))
sns.heatmap(df.corr(), annot = True)


# In[29]:


sns.set(style="ticks")
sns.pairplot(df, diag_kind = 'kde')


# In[30]:


df_1 = df['Ambient_Temperature']

df_2 = df[['Ambient_Temperature', 'Exhaust_Vaccum']]

df_3 = df[['Ambient_Temperature', 'Exhaust_Vaccum', 'Relative_Humidity']]

df_4 = df[['Ambient_Temperature', 'Exhaust_Vaccum', 'Relative_Humidity','Ambient_Pressure']]


# In[31]:


y = df['Energy_Output']


# In[32]:


X_train, X_test, y_train, y_test = train_test_split(df_4, y, test_size = 0.2, random_state = 0)


# In[33]:


dt_regressor= DecisionTreeRegressor()
dt_regressor.fit(X_train, y_train)
y_pred = dt_regressor.predict(X_test)


# In[34]:


print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('R Squared Value:', metrics.r2_score(y_test, y_pred))


# In[35]:


rf_regressor= RandomForestRegressor()
rf_regressor.fit(X_train, y_train)
y_pred = rf_regressor.predict(X_test)


# In[36]:


print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('R Squared Value:', metrics.r2_score(y_test, y_pred))

