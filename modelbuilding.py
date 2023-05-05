#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


# In[3]:


test=pd.read_csv('Dataset/loan-test.csv')
train=pd.read_csv('Dataset/loan-train.csv')


# In[4]:


train.head()


# In[5]:


plt.hist(np.log(train['LoanAmount']))


# In[6]:


test.head()


# In[7]:


train.head()


# In[8]:


train.isnull().sum()/train.shape[0]*100


# In[9]:


train.info()


# In[10]:


train['Gender'].fillna(train['Gender'].mode()[0],inplace=True)
train['Married'].fillna(train['Married'].mode()[0],inplace=True)
train['Dependents'].fillna(train['Dependents'].mode()[0],inplace=True)
train['Credit_History'].fillna(train['Credit_History'].mode()[0],inplace=True)
train['LoanAmount'].fillna(train['LoanAmount'].mean(),inplace=True)
train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mean(),inplace=True)


# In[11]:


test['Gender'].fillna(test['Gender'].mode()[0],inplace=True)
test['Married'].fillna(test['Married'].mode()[0],inplace=True)
test['Dependents'].fillna(test['Dependents'].mode()[0],inplace=True)
test['Credit_History'].fillna(test['Credit_History'].mode()[0],inplace=True)
test['LoanAmount'].fillna(test['LoanAmount'].mean(),inplace=True)
test['Loan_Amount_Term'].fillna(test['Loan_Amount_Term'].mean(),inplace=True)


# In[12]:


train=train.drop('Loan_ID',axis=1)
test=test.drop('Loan_ID',axis=1)


# In[13]:


from sklearn.preprocessing import LabelEncoder
lc=LabelEncoder()
label_encoder_column=['Gender','Education','Self_Employed','Property_Area','Loan_Status','Married']
for i in label_encoder_column:
    train[i]=lc.fit_transform(train[i])


# In[ ]:





# In[14]:


train.info()


# In[15]:


train.loc[train['Dependents']=='0','Dependents']=0
train.loc[train['Dependents']=='1','Dependents']=1
train.loc[train['Dependents']=='2','Dependents']=2
train.loc[train['Dependents']=='3+','Dependents']=3


# In[16]:


test.loc[test['Dependents']=='0','Dependents']=0
test.loc[test['Dependents']=='1','Dependents']=1
test.loc[test['Dependents']=='2','Dependents']=2
test.loc[test['Dependents']=='3+','Dependents']=3


# In[17]:


train.corr()


# In[18]:


plt.figure(figsize=(15,10))
sns.heatmap(train.corr(),annot=True,fmt=".2f",)
plt.show()


# In[19]:


y_train=train['Loan_Status']
x_train=train.drop(['Loan_Status','Loan_Amount_Term'],axis=1)


# In[20]:


x_train,x_test,y_train,y_test=train_test_split(x_train,y_train,test_size=0.3,random_state=0)


# In[21]:


test.head()


# In[22]:


lg=LogisticRegression()


# In[23]:


lg.fit(x_train,y_train)


# In[24]:


lg_predict=lg.predict(x_test)


# In[25]:


print(f'Accuracy percentage for LogisticRegression-{accuracy_score(y_test,lg_predict)*100}')


# In[27]:


import pickle


# In[28]:


pickle.dump(lg,open('LoanPrediction.pkl','wb'))


# In[ ]:



