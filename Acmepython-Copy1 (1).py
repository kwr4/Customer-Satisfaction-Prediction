#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd


# In[4]:


#The data sheet (from outlook)
df = pd.read_csv("ACME-HappinessSurvey2020 (1).csv")


# In[5]:


df


# In[6]:


dataset = pd.read_csv ("ACME-HappinessSurvey2020 (1).csv")


# In[7]:


#Head (start of the data)
dataset.head()


# In[8]:


#Tail(end of the data)
dataset.tail()


# In[9]:


X = dataset.drop(columns = 'Y')
y = dataset['Y']


# In[36]:


from sklearn.datasets import make_blobs

from sklearn.model_selection import train_test_split

X, y = make_blobs(n_samples=126)


# In[37]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, stratify = y)


# In[38]:


print(X)


# In[39]:


print(y)


# In[40]:


print(X_train)


# In[41]:


print(X_test)


# In[42]:


print(y_train)


# In[43]:


print(y_test)


# In[150]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)


# In[47]:


df = pd.read_csv("ACME-HappinessSurvey2020 (1).csv")
df.head()


# In[145]:


print(df.shape)


print(df.columns)

print(df.info())


# In[51]:


# Description of statistical information
df.describe()


# In[52]:


df["X1"].mean()


# In[53]:


df.sort_values(by="X1", ascending=False).head()


# In[54]:


import seaborn as sns                       #visualisation
import matplotlib.pyplot as plt             #visualisation
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(color_codes=True)


# In[55]:


df.dtypes


# In[57]:


df = df.rename(columns={"Score": "Y"})


# In[59]:


sns.boxplot(x=df['X1'])


# In[60]:


sns.boxplot(x=df['X2'])


# In[61]:


sns.boxplot(x=df['X3'])


# In[62]:


sns.boxplot(x=df['X4'])


# In[63]:


sns.boxplot(x=df['X5'])


# In[64]:


sns.boxplot(x=df['X6'])


# In[65]:


#Heatmap
plt.figure(figsize=(10,5))
c= df.corr()
sns.heatmap(c,cmap="BrBG",annot=True)
c


# In[66]:


import matplotlib.pyplot as plt
import numpy as np


# In[152]:


s=df['Y']
print(s.head())

df["Y"].value_counts()

df["Y"].mean()


# In[157]:


df["X1"].value_counts()


# In[156]:


df["X1"].mean()


# In[72]:


df["X2"].value_counts()


# In[73]:


df["X2"].mean()


# In[74]:


df["X3"].value_counts()


# In[75]:


df["X3"].mean()


# In[76]:


df["X4"].value_counts()


# In[77]:


df["X4"].mean()


# In[78]:


df["X5"].value_counts()


# In[79]:


df["X5"].mean()


# In[80]:


df["X6"].mean()


# In[81]:


df["X6"].value_counts()


# In[82]:


#creating happy =1 and sad = 0. This lets us seperate between happy and sad.
df_happy = df[df['Y'] == 1]
df_sad = df[df['Y'] == 0]


# In[161]:


df_happy.head()


# In[160]:


df_sad.head()


# In[85]:


df.columns


# In[86]:


df_happy[['X1', 'X2', 'X3', 'X4', 'X5', 'X6']].mean()


# In[87]:


df_sad[['X1', 'X2', 'X3', 'X4', 'X5', 'X6']].mean(axis = 0)


# In[88]:


df_happy[['X1', 'X2', 'X3', 'X4', 'X5', 'X6']].mean(axis = 1)


# In[163]:


df_happy[['X1', 'X2']].plot()

df_happy[['X1', 'X3']].plot()

df_happy[['X1', 'X4']].plot()

df_happy[['X1', 'X5']].plot()

df_happy[['X1', 'X6']].plot()

df_sad[['X1', 'X2']].plot()

df_sad[['X1', 'X3']].plot()

df_sad[['X1', 'X4']].plot()

df_sad[['X1', 'X5']].plot()

df_sad[['X1', 'X6']].plot()


# In[103]:


#Getting ready to start the algorithm and defining y and x as well as train and text/split.
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

Y = df[['Y']]
X = df[['X1', 'X2', 'X3', 'X4', 'X5', 'X6']]

logReg = LogisticRegression()

x_train, x_test, y_train, y_test = train_test_split(X, Y)


# In[104]:


print(x_train.shape)
print(x_test.shape)


# In[105]:


logReg.fit(x_train, y_train)


# In[106]:


logReg.predict(x_test)


# In[107]:


x_test['predictions_from_model'] = logReg.predict(x_test)
x_test.head()


# In[109]:


x_test['pred_probabilities_from_model'] = logReg.predict_proba(x_test[['X1', 'X2', 'X3', 'X4', 'X5', 'X6']])[:,1]


# In[110]:


logReg.predict_proba(x_test[['X1', 'X2', 'X3', 'X4', 'X5', 'X6']])[:,1]


# In[119]:


x_test


# In[195]:


#Importing these to get ready for performance metrics
import pandas
from sklearn.metrics import accuracy_score
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[179]:


# Cross Validation Classification Accuracy
dataframe = pd.read_csv("ACME-HappinessSurvey2020 (1).csv")
array = dataframe.values
X = array[:,0:6]
Y = array[:,1]
kfold = model_selection.KFold(n_splits=10, random_state=14, shuffle=True)
model = LogisticRegression(solver='liblinear')
scoring = 'accuracy'
results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("Accuracy: %.3f (%.3f)" % (results.mean(), results.std()))


# In[181]:


# Cross Validation Classification LogLoss
dataframe = pd.read_csv("ACME-HappinessSurvey2020 (1).csv")
array = dataframe.values
X = array[:,0:6]
Y = array[:,1]
kfold = model_selection.KFold(n_splits=10, random_state=14, shuffle=True)
model = LogisticRegression(solver='liblinear')
scoring = 'neg_log_loss'
results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("Logloss: %.3f (%.3f)" % (results.mean(), results.std()))


# In[182]:


# Cross Validation Classification ROC AUC
dataframe = pd.read_csv("ACME-HappinessSurvey2020 (1).csv")
array = dataframe.values
X = array[:,0:6]
Y = array[:,1]
kfold = model_selection.KFold(n_splits=10, random_state=14, shuffle=True)
model = LogisticRegression(solver='liblinear')
scoring = 'roc_auc'
results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("AUC: %.3f (%.3f)" % (results.mean(), results.std()))


# In[183]:


# Cross Validation Classification Confusion Matrix
dataframe = pd.read_csv("ACME-HappinessSurvey2020 (1).csv")
array = dataframe.values
X = array[:,0:6]
Y = array[:,1]
test_size = 0.33

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=7)

model = LogisticRegression(solver='liblinear')

model.fit(X_train, Y_train)

predicted = model.predict(X_test)

matrix = confusion_matrix(Y_test, predicted)
print(matrix)


# In[203]:


# Cross Validation Classification Report
dataframe = pd.read_csv("ACME-HappinessSurvey2020 (1).csv")

array = dataframe.values

X = array[:,0:6]

Y = array[:,1]

test_size = 0.33

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=7)

model = LogisticRegression(solver='liblinear')


model.fit(X_train, Y_train)

predicted = model.predict(X_test)

report = classification_report(Y_test, predicted)

print(report)


# In[200]:


# Cross Validation Regression MAE
dataframe = pd.read_csv("ACME-HappinessSurvey2020 (1).csv")
array = dataframe.values
X = array[:,0:6]
Y = array[:,1]
kfold = model_selection.KFold(n_splits=10, random_state=14, shuffle=True)
model = LinearRegression()
scoring = 'neg_mean_absolute_error'
results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("MAE: %.3f (%.3f)" % (results.mean(), results.std()))


# In[201]:


# Cross Validation Regression MSE
dataframe = pd.read_csv("ACME-HappinessSurvey2020 (1).csv")
array = dataframe.values
X = array[:,0:6]
Y = array[:,1]
kfold = model_selection.KFold(n_splits=10, random_state=14, shuffle=True)
model = LinearRegression()
scoring = 'neg_mean_squared_error'
results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("MSE: %.3f (%.3f)" % (results.mean(), results.std()))


# In[202]:


# Cross Validation Regression R^2
dataframe = pd.read_csv("ACME-HappinessSurvey2020 (1).csv")
array = dataframe.values
X = array[:,0:6]
Y = array[:,1]
kfold = model_selection.KFold(n_splits=10, random_state=14, shuffle=True)
model = LinearRegression()
scoring = 'r2'
results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("R^2: %.3f (%.3f)" % (results.mean(), results.std()))


# In[ ]:




