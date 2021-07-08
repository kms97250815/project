#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling
import os
import scipy.stats as st
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import confusion_matrix, classification_report 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')


# In[3]:


data = pd.read_csv("heart.csv") 


# 기술통계

# In[4]:


data.describe()


# In[5]:


data.head(10)


# In[6]:


data.info()


# In[7]:


profile = pandas_profiling.ProfileReport(data)
profile


# Viualization

# In[7]:


fig, axis = plt.subplots(7,2,figsize=(15,20))
data.hist(ax=axis)
plt.show() 


# In[8]:


columns=['age','trestbps','chol','thalach','oldpeak','target']
sns.pairplot(data[columns],hue='target',corner=True,diag_kind='hist')


# In[9]:


f=sns.countplot(x='sex',data=data)
sns.countplot(data.sex)
f.set_title("Gender Distribution")
f.set_xticklabels(["Female","Male"])
plt.show()


# In[10]:


total_genders_count=len(data.sex)
male_count=len(data[data['sex']==1])
female_count=len(data[data['sex']==0])
print('Total Genders :',total_genders_count)
print('Male Count    :',male_count)
print('Female Count  :',female_count)


# In[11]:


e=plt.figure(figsize=(13,8))
f=sns.countplot(x='age', data=data)
f.set_title("Age Distribution")
plt.show()


# In[12]:


plt.figure(figsize=(14,5))
plt.title("Age/Heart Disease Relationship")
plt.xlabel("Age")
plt.ylabel("Count")

yt = plt.hist(data[data['target']==0]['age'], alpha = 0.5, bins = 50,
             label = 'Heart Disease')
nt = plt.hist(data[data['target']==1]['age'], color = 'orange', alpha = 0.5, bins = 50,
             label = 'No Heart Disease')
plt.legend(["Female","Male"])


# In[13]:


minAge=min(data.age)
maxAge=max(data.age)
meanAge=data.age.mean()
print('Min Age :',minAge)
print('Max Age :',maxAge)
print('Mean Age :',meanAge)


# In[14]:


total_target = len(data.target)
target_count = len(data[data['target']==1])
nontarget_count = len(data[data['target']==0])
print('Population:',total_target)
print('No Heart Disease:',nontarget_count)
print('Heart Disease:',target_count)


# In[15]:


f=sns.countplot(x='target',data=data)
f.set_title("Heart disease presence distribution")
f.set_xticklabels(["No heart disease","Heart Disease"])
plt.xlabel("") ##Heart disease precense distribution


# In[16]:


f=sns.countplot(x='target',data=data,hue='sex')
plt.legend(["Female","Male"])
f.set_title("Heart disease presence by gender")
f.set_xticklabels(["No heart disease","Heart Disease"])
plt.xlabel("") 


# In[17]:


f=sns.countplot(x='target',data=data,hue='sex')
plt.legend(["Female","Male"])
f.set_title("Heart disease presence by gender")
f.set_xticklabels(["No heart disease","Heart Disease"])
plt.xlabel("") 


# In[18]:


f=sns.countplot(x='cp',data=data,hue='target')
f.set_title("Chest Pain/Heart Disease Relationship")
plt.legend(labels=['No Heart Disease', 'Heart Disease'])
plt.xlabel("")


# In[19]:


f=sns.countplot(x='slope',data=data,hue='target')
f.set_title("Heart Disease Frequency for Slope")
plt.legend(labels=['No Heart Disease', 'Heart Disease'])
plt.xlabel("")


# In[20]:


f=sns.countplot(x='fbs',data=data,hue='target')
f.set_title("FBS/Heart Disease Relationship.FBS > 120 mg/dl")
plt.legend(labels=['No Heart Disease', 'Heart Disease'])
plt.xlabel("")


# In[21]:


plt.figure(figsize=(15,5))
f=sns.lineplot(x="age", y="trestbps", hue="target", data=data)
f.set_title("Blood Pressure/Heart Disease Relationship")
plt.legend(labels=['No Heart Disease', 'Heart Disease'])


상관관계
# In[22]:


plt.figure(figsize=(15,10))
f=sns.heatmap(data.corr(),cmap='Blues',annot=True)
f.set_title("Overall Correlation")


# In[23]:


plt.figure(figsize=(15,10))
f=sns.heatmap(data[data['target']==0].corr(),cmap='Blues',annot=True)
f.set_title("No Heart Disease Correlation")


# In[24]:


plt.figure(figsize=(15,10))
f=sns.heatmap(data[data['target']==1].corr(),cmap='Blues',annot=True)
f.set_title("Heart Disease Correlation")


# In[25]:


data.sex=data.sex.astype('category')
data.cp=data.cp.astype('category')
data.fbs=data.fbs.astype('category')
data.restecg=data.restecg.astype('category')
data.exang=data.exang.astype('category')
data.ca=data.ca.astype('category')
data.slope=data.slope.astype('category')
data.thal=data.thal.astype('category')


# In[26]:


data_label=data['target']
del data['target']
data_label=pd.DataFrame(data_label)


# In[27]:


data=pd.get_dummies(data,drop_first=True)
data.head(),data_label.head()


# In[28]:


data_scaled=MinMaxScaler().fit_transform(data)
data_scaled=pd.DataFrame(data=data_scaled, columns=data.columns)


# In[29]:


def CrossVal(dataX,dataY,mode,cv=3):
    score=cross_val_score(mode,dataX , dataY, cv=cv, scoring='accuracy')
    return(np.mean(score))


# In[30]:


Xtrain,Xtest,Ytrain,Ytest = train_test_split(data_scaled, data_label, test_size=0.20,
                                             stratify=data_label,random_state=9154)


# In[31]:


def plotting(true,pred):
    fig,ax=plt.subplots(1,2,figsize=(10,5))
    precision,recall,threshold = precision_recall_curve(true,pred[:,1])
    ax[0].plot(recall,precision,'g--')
    ax[0].set_xlabel('Recall')
    ax[0].set_ylabel('Precision')
    ax[0].set_title("Average Precision Score : {}".format(average_precision_score(true,pred[:,1])))
    fpr,tpr,threshold = roc_curve(true,pred[:,1])
    ax[1].plot(fpr,tpr)
    ax[1].set_title("AUC Score is: {}".format(auc(fpr,tpr)))
    ax[1].plot([0,1],[0,1],'k--')
    ax[1].set_xlabel('False Positive Rate')
    ax[1].set_ylabel('True Positive Rate')


# In[32]:


k=KNeighborsClassifier(algorithm='auto',n_neighbors= 19)
score_k=CrossVal(Xtrain,Ytrain,k)
print("Accuracy is : ",score_k)
k.fit(Xtrain,Ytrain)
plotting(Ytest,k.predict_proba(Xtest))


fig=plt.figure()
sns.heatmap(confusion_matrix(Ytest,k.predict(Xtest)), annot= True, cmap='Reds')
k_f1=f1_score(Ytest,k.predict(Xtest))
plt.title('F1 Score = {}'.format(k_f1))


# In[33]:


lr=LogisticRegression(class_weight='balanced', tol=1e-10)
score_lr=CrossVal(Xtrain,Ytrain,lr)
print("Accuracy is : ",score_lr)
lr.fit(Xtrain,Ytrain)
plotting(Ytest,lr.predict_proba(Xtest))


fig=plt.figure()
sns.heatmap(confusion_matrix(Ytest,lr.predict(Xtest)), annot= True, cmap='Greens')
lr_f1=f1_score(Ytest,lr.predict(Xtest))
plt.title('F1 Score = {}'.format(lr_f1))


# In[34]:


dtc=DecisionTreeClassifier(max_depth=6)
score_dtc=CrossVal(Xtrain,Ytrain,dtc)
print("Accuracy is : ",score_dtc)
dtc.fit(Xtrain,Ytrain)
plotting(Ytest,dtc.predict_proba(Xtest))

fig=plt.figure()
sns.heatmap(confusion_matrix(Ytest,dtc.predict(Xtest)), annot= True, cmap='Blues')

dtc_f1=f1_score(Ytest,dtc.predict(Xtest))
plt.title('F1 Score = {}'.format(dtc_f1))


# In[35]:


svc=SVC(C=0.2,probability=True,kernel='rbf',gamma=0.1)
score_svc=CrossVal(Xtrain,Ytrain,svc)
print("Accuracy is : ",score_svc)
svc.fit(Xtrain,Ytrain)
plotting(Ytest,svc.predict_proba(Xtest))

fig=plt.figure()
sns.heatmap(confusion_matrix(Ytest,svc.predict(Xtest)), annot= True, cmap='Greys')
svc_f1=f1_score(Ytest,svc.predict(Xtest))
plt.title('F1 Score = {}'.format(svc_f1))


# In[37]:


rf=RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=97)
score_rf= CrossVal(Xtrain,Ytrain,rf)
print('Accuracy is:',score_rf)
rf.fit(Xtrain,Ytrain)
plotting(Ytest,rf.predict_proba(Xtest))

fig=plt.figure()
sns.heatmap(confusion_matrix(Ytest,rf.predict(Xtest)), annot= True, cmap='Oranges')

rf_f1=f1_score(Ytest,rf.predict(Xtest))
plt.title('F1 Score = {}'.format(rf_f1))


# In[49]:


model_accuracy = pd.Series(data=[score_k, score_lr, score_dtc, score_svc, score_rf], 
                           index=['KNN','logistic Regression','decision tree', 'SVM', 'Random Forest'])
fig= plt.figure(figsize=(8,8))
model_accuracy.sort_values().plot.barh()
plt.title('Model Accuracy')

print("Logistic Regression's accuracy is : ",score_lr)
print("KNN's accuracy is : ",score_k)
print("SVM's accuracy is : ",score_svc)
print("Random Forest's accuracy is: ",score_rf)
print("Decision Tree's accuracy is : ",score_dtc)





# In[48]:


model_f1_score = pd.Series(data=[k_f1, lr_f1, dtc_f1, svc_f1, rf_f1], 
                           index=['KNN','logistic Regression','decision tree', 'SVM', 'Random Forest'])
fig= plt.figure(figsize=(8,8))
model_f1_score.sort_values().plot.barh()
plt.title('F1 Score Comparison')

print("KNN's F1 score is : ",k_f1)
print("Logistic Regression's F1 score is : ",lr_f1)
print("SVM's F1 score is : ",svc_f1)
print("Random Forest's F1 score is: ",rf_f1)
print("Decision Tree's F1 score is : ",dtc_f1)


