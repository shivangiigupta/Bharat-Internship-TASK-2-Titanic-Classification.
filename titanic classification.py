#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv("tested.csv")
df.head()


# In[3]:


df.head(10)


# In[ ]:


#EDA


# In[4]:


df.describe()


# In[5]:


df.shape


# In[6]:


df['Survived'].value_counts()


# In[7]:


#let's visualize the count of survivals wrt pclass
sns.countplot(x=df['Survived'], hue=df['Pclass'])


# In[8]:


df["Sex"]


# In[9]:


#let's visualize the count of survivals wrt Gender
sns.countplot(x=df['Sex'], hue=df['Survived'])


# In[10]:


df.mean()


# In[11]:


#Look at survival rate by sex
df.groupby('Sex')[['Survived']].mean()


# In[ ]:





# In[12]:


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

df['Sex']= labelencoder.fit_transform(df['Sex'])

df.head()


# In[13]:


df['Sex'], df['Survived']


# In[14]:


sns.countplot(x=df['Sex'], hue=df["Survived"])


# In[15]:


df.isna().sum()


# In[29]:


for i in ["Pclass","Sex","Embarked"]:
    print(i,df[i].value_counts())


# In[33]:


df.loc[(df["Sex"]==0) & (df['Fare']<50)] 


# In[16]:


# After dropping non required column
df=df.drop(['Age'], axis=1)
df_final = df
df_final.head(10)


# In[17]:


X= df[['Pclass', 'Sex']]
Y=df['Survived']


# In[18]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


# In[19]:


from sklearn.linear_model import LogisticRegression

log = LogisticRegression(random_state = 0)
log.fit(X_train, Y_train)


# In[20]:


pred = print(log.predict(X_test))


# In[21]:


print(Y_test)


# In[27]:


import warnings
warnings.filterwarnings("ignore")

res= log.predict([[1,1]])

if(res==0):
  print("So Sorry! Not Survived")
else:
  print("Survived")


# In[28]:


import warnings
warnings.filterwarnings("ignore")

res= log.predict([[0,0]])

if(res==0):
  print("So Sorry! Not Survived")
else:
  print("Survived")


# In[ ]:





# In[ ]:





# In[ ]:




