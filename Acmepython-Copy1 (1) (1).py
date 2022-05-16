#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


#The data sheet (from outlook)
df = pd.read_csv("ACME-HappinessSurvey2020 (1).csv")


# In[3]:


df


# In[4]:


dataset = pd.read_csv ("ACME-HappinessSurvey2020 (1).csv")


# In[5]:


#Head (start of the data)
dataset.head()


# In[6]:


#Tail(end of the data)
dataset.tail()


# In[7]:


X = dataset.drop(columns = 'Y')
y = dataset['Y']


# In[8]:


from sklearn.datasets import make_blobs

from sklearn.model_selection import train_test_split

X, y = make_blobs(n_samples=126)


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, stratify = y)


# In[10]:


print(y)


# In[11]:


print(y_train)


# In[12]:


print(y_test)


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)


# In[14]:


df = pd.read_csv("ACME-HappinessSurvey2020 (1).csv")
df.head()


# In[15]:


print(df.shape)


print(df.columns)

print(df.info())


# In[16]:


# Description of statistical information
# y mean = 54.7% , this means that slightly more customers are happy than sad

# x1 mean = 4.3   x2 mean = 2.5   x3 mean = 3.3  x5 mean = 3.6    x6 mean = 4.2

# x1 top 75% = 5.0, 25% = 4.0   This means that x1 scores are mostly between 4 and 5
df.describe()


# In[17]:


df.sort_values(by="X1", ascending=False).head()


# In[18]:


import seaborn as sns                       #visualisation
import matplotlib.pyplot as plt             #visualisation
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(color_codes=True)


# In[19]:


df = df.rename(columns={"Score": "Y"})


# In[20]:


#X1 boxplot showing a score leaning towards 4.0 and 5.0
sns.boxplot(x=df['X1'])


# In[21]:


#x2 boxplot showing scores between 2.0 and 3.0
sns.boxplot(x=df['X2'])


# In[22]:


#x3 boxplot showing score between 3.0 and 4.0
sns.boxplot(x=df['X3'])


# In[23]:


#x4 boxplot showing scores between 3.0 and 4.0
sns.boxplot(x=df['X4'])


# In[24]:


#x5 boxplot showing scores between 3.0 and 4.0
sns.boxplot(x=df['X5'])


# In[25]:


#x6 boxplot showing scores between 4.0 and 5.0. This boxplot has an outlier of 2.0.
sns.boxplot(x=df['X6'])


# In[26]:


#Heatmap, relations between variables

#x1 to x2 = .06 correlation       #x2 to x3 =  .18     #x3 to x4 =   .3      #x4 to x5 = .29        #x5 to x6 = .32     
#x1 to x3=  .28 correlation       #x2 to x4 =  .11     #x3 to x5 =   .36     #x4 to x6 = .22                                           
#x1 to x4=  .088 correlation      #x2 to x5 =  .04     #x3 to x6 =   .2                             
#x1 to x5=  .43 correlation       #x2 to x6 = -.062    
#x1 to x6=  .41 correlation        


plt.figure(figsize=(10,5))
c= df.corr()
sns.heatmap(c,cmap="BrBG",annot=True)
c


# In[27]:


import matplotlib.pyplot as plt
import numpy as np


# In[28]:


s=df['Y']
print(s.head())

df["Y"].value_counts()

df["Y"].mean()


# In[29]:


df["X1"].value_counts()


# In[30]:


df["X2"].value_counts()


# In[31]:


df["X3"].value_counts()


# In[32]:


df["X4"].value_counts()


# In[33]:


df["X5"].value_counts()


# In[34]:


df["X6"].value_counts()


# In[35]:


#creating happy =1 and sad = 0. This lets us seperate between happy and sad.
df_happy = df[df['Y'] == 1]
df_sad = df[df['Y'] == 0]


# In[36]:


df_happy.head()


# In[37]:


df_sad.head()


# In[38]:


df.columns


# In[39]:


df_happy[['X1', 'X2', 'X3', 'X4', 'X5', 'X6']].mean()


# In[40]:


df_sad[['X1', 'X2', 'X3', 'X4', 'X5', 'X6']].mean(axis = 0)


# In[41]:


df_happy[['X1', 'X2', 'X3', 'X4', 'X5', 'X6']].mean(axis = 1)


# In[42]:


#Line Graph showing correlations between different x variables
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


# In[43]:


#Getting ready to start the algorithm and defining y and x as well as train and text/split.
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

Y = df[['Y']]
X = df[['X1', 'X2', 'X3', 'X4', 'X5', 'X6']]

logReg = LogisticRegression()

x_train, x_test, y_train, y_test = train_test_split(X, Y)


# In[44]:


print(x_train.shape)
print(x_test.shape)


# In[45]:


logReg.fit(x_train, y_train)


# In[46]:


logReg.predict(x_test)


# In[47]:


x_test['predictions_from_model'] = logReg.predict(x_test)
x_test.head()


# In[48]:


x_test['pred_probabilities_from_model'] = logReg.predict_proba(x_test[['X1', 'X2', 'X3', 'X4', 'X5', 'X6']])[:,1]


# In[49]:


logReg.predict_proba(x_test[['X1', 'X2', 'X3', 'X4', 'X5', 'X6']])[:,1]


# In[50]:


x_test


# In[51]:


#Importing these to get ready for performance metrics
import pandas
from sklearn.metrics import accuracy_score
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LinearRegression


# In[52]:


# Cross Validation Classification Accuracy
dataframe = pd.read_csv("ACME-HappinessSurvey2020 (1).csv")
array = dataframe.values
X = array[:,1:6]
Y = array[:,0:1]
kfold = model_selection.KFold(n_splits=10, random_state=10, shuffle=True)
model = LogisticRegression(solver='liblinear')
scoring = 'accuracy'
results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("Accuracy: %.3f (%.3f)" % (results.mean(), results.std()))


# In[53]:


# Cross Validation Classification LogLoss
dataframe = pd.read_csv("ACME-HappinessSurvey2020 (1).csv")
array = dataframe.values
X = array[:,1:6]
Y = array[:,0:1]
kfold = model_selection.KFold(n_splits=10, random_state=7, shuffle=True)
model = LogisticRegression(solver='liblinear')
scoring = 'neg_log_loss'
results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("Logloss: %.3f (%.3f)" % (results.mean(), results.std()))


# In[54]:


# Cross Validation Classification ROC AUC
dataframe = pd.read_csv("ACME-HappinessSurvey2020 (1).csv")
array = dataframe.values
X = array[:,1:6]
Y = array[:,0:1]
kfold = model_selection.KFold(n_splits=10, random_state=7, shuffle=True)
model = LogisticRegression(solver='liblinear')
scoring = 'roc_auc'
results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("AUC: %.3f (%.3f)" % (results.mean(), results.std()))


# In[55]:


# Cross Validation Classification Confusion Matrix
dataframe = pd.read_csv("ACME-HappinessSurvey2020 (1).csv")
array = dataframe.values
X = array[:,1:6]
Y = array[:,0:1]
test_size = 0.33

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=7)

model = LogisticRegression(solver='liblinear')

model.fit(X_train, Y_train)

predicted = model.predict(X_test)

matrix = confusion_matrix(Y_test, predicted)
print(matrix)


# In[56]:


# Cross Validation Classification Report
dataframe = pd.read_csv("ACME-HappinessSurvey2020 (1).csv")

array = dataframe.values

X = array[:,1:6]

Y = array[:,0:1]

test_size = 0.33

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=7)

model = LogisticRegression(solver='liblinear')


model.fit(X_train, Y_train)

predicted = model.predict(X_test)

report = classification_report(Y_test, predicted)

print(report)


# In[57]:


# Cross Validation Regression MAE
dataframe = pd.read_csv("ACME-HappinessSurvey2020 (1).csv")
array = dataframe.values
X = array[:,1:6]
Y = array[:,0:1]
kfold = model_selection.KFold(n_splits=10, random_state=14, shuffle=True)
model = LinearRegression()
scoring = 'neg_mean_absolute_error'
results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("MAE: %.3f (%.3f)" % (results.mean(), results.std()))


# In[58]:


# Cross Validation Regression MSE
dataframe = pd.read_csv("ACME-HappinessSurvey2020 (1).csv")
array = dataframe.values
X = array[:,1:6]
Y = array[:,0:1]
kfold = model_selection.KFold(n_splits=10, random_state=14, shuffle=True)
model = LinearRegression()
scoring = 'neg_mean_squared_error'
results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("MSE: %.3f (%.3f)" % (results.mean(), results.std()))


# In[59]:


# Cross Validation Regression R^2
dataframe = pd.read_csv("ACME-HappinessSurvey2020 (1).csv")
array = dataframe.values
X = array[:,1:6]
Y = array[:,0:1]
kfold = model_selection.KFold(n_splits=10, random_state=14, shuffle=True)
model = LinearRegression()
scoring = 'r2'
results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("R^2: %.3f (%.3f)" % (results.mean(), results.std()))


# In[60]:


from scipy.stats import pearsonr
data1 = 
data2 = 
stat, p = pearsonr(data1, data2)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')


# In[ ]:


# Example of the Chi-Squared Test
from scipy.stats import chi2_contingency
table = [[],[]]
stat, p, dof, expected = chi2_contingency(table)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')


# In[ ]:





# In[ ]:


#Attempting Naive Bayes
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[ ]:


dataset = pd.read_csv ("ACME-HappinessSurvey2020 (1).csv")
X = dataset.iloc[:,0:1].values
y = dataset.iloc[:,0:1].values


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[ ]:


from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)


# In[ ]:


y_pred  =  classifier.predict(X_test)


# In[ ]:


y_pred  


# In[ ]:


from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test,y_pred)


# In[ ]:


from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)


# In[ ]:





# In[ ]:


#Attempting KNN
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)


# In[ ]:


y_pred = classifier.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[ ]:


#check

    "1. Import the data and do exploratory data analysis",
    #will recheck to make sure I understand

    "2. Check for NULLs/NAs and outliers",    
    # do not see any. I do see an outlier in x6 where it is marked on the 2.0
    

    "3. If NULLs/NAs are found, check if you can impute any values depending on the variable type. Or if you can safely remove it, remove them",
    # n/a

    "4. If Outliers are found, check how far they are from the mean or median and check if such data makes sense to have such value",
    # yes it does make sense as the range can be from 1.0 - 5.0
    
    "5. If the data value makes sense, then exclude those records and keep them aside for sub-analysis and if the data doesn't make sense, either impute them with logical value or the highest possible value that makes sense",
  # checked  
    
    "6. Check for data imbalance. If the data is found to be imbalanced then we'd have to do oversampling/undersampling of training data for minority class",
 # unsure for this number as i believe it was overfit. will recheck again.    
    
    "7. Do necessary cosmetic changes and run a significance testing between each independent and dependent variable. This will tell us the IVs (Independent Variables) which can explain our DV (Dependent Variable)",
    # in process, researching how to input this data on python
    
    "8. Run a multivariate analysis between all IVs to see which of it can be removed for high correlation value",
    # in process, researching how to input this data on python
    
    "9. Check if the data makes sense to run a logistic regression on",
     # yes, data makes sense to run logistic regresson on because we are attempting to predict a category/classify. binary outcomes. output is 0 or 1. 
    
    "10. Once it is compatible, run a logistic regression on the entire data to check how well each IV explains DV when all IVs are present together",
   #in process, still researching
      

    "11. Identify right features from all these cases and divide the data into train and test",
    #in process, making sure I am doing it correct.
    
    "12. If the data was found to be imbalance, use either random or SMOTE oversampling or random undersampling on training data",
    #still testing. 
    
    "13. Run logistic regression and optimize it and get the best accuracy, recall, precision and F1 scores for class 0 and class 1 and overall",
    precision    recall  f1-score   support

           0       0.70      0.44      0.54        16
           1       0.72      0.88      0.79        26

    accuracy                           0.71        42
   macro avg       0.71      0.66      0.67        42
weighted avg       0.71      0.71      0.70        42
    
    "13. Run Naive Bayes",
    #in process, attempted naive
    
    "14. Run kNN, SVM",
    
    #attempted knn
    "15. Run Random Forest & XGBoost",
    
    
    "16. Do hyperparameter tuning for all these models, using GridSearchCV"
    
    
    
    "17. Pick the best model and summarize the best results"


# In[ ]:





# In[ ]:




