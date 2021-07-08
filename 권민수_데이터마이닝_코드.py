#!/usr/bin/env python
# coding: utf-8

# In[138]:


import numpy as np
import pandas as pd
import math
import numpy as np
import missingno as msno
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
plt.style.use('seaborn-darkgrid')
palette = plt.get_cmap('Set2')
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense,Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,roc_auc_score,roc_curve
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from collections import Counter
import warnings
warnings.filterwarnings("ignore")


# In[90]:


df = pd.read_csv("telecom_users.csv") 


# In[112]:


dataset = pd.read_csv("telecom_users.csv") 


# In[44]:


df.head()


# In[5]:


df.nunique()


# In[124]:


df.describe()


# In[62]:


df.info()


# In[65]:


df.isnull()


# In[67]:


[col for col in df.columns if " " in df[col].value_counts().index.tolist()]


# In[16]:


## VISUALIZATION


# In[18]:


gender=df["gender"].value_counts()
plt.figure(figsize=(8,6))
plot=gender.plot.pie(autopct="%1.0f%%")
plt.show()


# In[34]:


ismarried=df['Partner'].value_counts()
plt.title('Clients Married or Not')
plt.ylabel('Counts')
sns.barplot(x=ismarried.index,y=ismarried.values)
plt.show()


# In[35]:


tenure = df['tenure']
plt.figure(figsize=(10,8))
sns.histplot(tenure, bins=50, alpha=0.8)
plt.xticks(list(range(0,tenure.max(),5)))
plt.yticks(list(range(0,550,50)))
plt.title('Client Service Usage')
plt.show()


# In[22]:


other_servises = ['MultipleLines', 'InternetService', 'OnlineSecurity',
                  'OnlineBackup', 'DeviceProtection', 'TechSupport',
                  'StreamingTV', 'StreamingMovies']

fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(20, 15), sharey=True)
t = 0

for i in range(4):
    for j in range(2):
        data_prep = [df[obj].value_counts()  for obj in other_servises]
        axes = ax[i][j]

        sns.barplot(x=data_prep[t].index, y=data_prep[t].values, ax=axes, alpha=0.8)
        axes.set_title(other_servises[t], fontsize=16)
        axes.set_yticks(list(range(0, 3001, 500)))
        if j == 0:
            axes.set_ylabel('Number of clients', fontsize=12)
        t += 1

fig.suptitle('Customers who use additional services', y=0.93, fontsize=20)
plt.show()


# In[23]:


contract=df['Contract'].value_counts()
plt.figure(figsize=(10,8))
sns.barplot(y=contract.values,x=contract.index)
plt.title('Contracts vs Number of clients',size=20)
plt.show()


# In[25]:


billing=df['PaperlessBilling'].value_counts()
plt.figure(figsize=(8,6))
sns.barplot(x=billing.index,y=billing.values)
plt.title('Paperless Biling',size=20)
plt.show()


# In[39]:


payment=df['PaymentMethod'].value_counts()
plt.figure(figsize=(10,8))
sns.barplot(x=payment.index,y=payment.values)
plt.title("Clients' method of payment",size=20)
plt.ylabel('No of clients',size=20)
plt.xlabel('Payment method',size=20)
plt.show()


# In[10]:


charges=df['MonthlyCharges']
plt.figure(figsize=(10,8))
sns.distplot(charges)
plt.title('Payment Density')
plt.xlabel('Monthly Charges')
plt.show()


# In[28]:


depends=df['Dependents'].value_counts()
plt.figure(figsize=(10,8))
sns.barplot(x=depends.index,y=depends.values)
plt.ylabel('No of Clients')
plt.yticks(list(range(0,4500,100)))
plt.show()


# In[29]:


plt.figure(figsize=(20,10))
streamingtv=df['StreamingTV'].value_counts()
plt.yticks(list(range(0,2500,200)))
sns.barplot(x=streamingtv.index,y=streamingtv.values)
plt.title('Clients streaming TV')
plt.ylabel(' No of Clients streaming TV')
plt.show()


# In[30]:


streamingmov=df['StreamingMovies'].value_counts()
plt.figure(figsize=(20,10))
plt.yticks(list(range(0,2500,100)))
sns.barplot(x=streamingmov.index,y=streamingmov.values)
plt.title('Clients streaming Movies')
plt.ylabel(' No of Clients streaming Movies')
plt.show()


# In[11]:


techsupport=df["TechSupport"].value_counts()
plt.figure(figsize=(8,6))
plot=techsupport.plot.pie(autopct="%1.0f%%",frame=True,shadow=True)
plt.title('Gets Tech Support')
plt.show()


# In[50]:


plt.figure(figsize=(8,8))
churn=df['Churn']
plt.yticks(list(range(0,5000,500)))
sns.histplot(churn,binwidth=20)


# In[111]:


for col in data_copy.columns[2:]:
    print(f"{col}: \n{data_copy[col].value_counts()}\n\n")


# In[114]:


dataset['Churn']=dataset['Churn'].replace(to_replace=0,value='No')
dataset['Churn']=dataset['Churn'].replace(to_replace=1,value='Yes')


# In[115]:


dataset['Churn'].value_counts()


# In[116]:


dataset['MultipleLines']=dataset['MultipleLines'].replace(to_replace='No internet service',value='No')
dataset['OnlineSecurity']=dataset['OnlineSecurity'].replace(to_replace='No internet service',value='No')
dataset['OnlineBackup']=dataset['OnlineBackup'].replace(to_replace='No internet service',value='No')
dataset['DeviceProtection']=dataset['DeviceProtection'].replace(to_replace='No internet service',value='No')
dataset['TechSupport']=dataset['TechSupport'].replace(to_replace='No internet service',value='No')
dataset['StreamingTV']=dataset['StreamingTV'].replace(to_replace='No internet service',value='No')
dataset['StreamingMovies']=dataset['StreamingMovies'].replace(to_replace='No internet service',value='No')
dataset


# In[117]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
dataset['gender']=le.fit_transform(dataset['gender'])
dataset['Churn']=le.fit_transform(dataset['Churn'])
dataset['PhoneService']=le.fit_transform(dataset['PhoneService'])
dataset['PaperlessBilling']=le.fit_transform(dataset['PaperlessBilling'])
dataset['Dependents']=le.fit_transform(dataset['Dependents'])
dataset['Partner']=le.fit_transform(dataset['Partner'])
dataset['MultipleLines']=le.fit_transform(dataset['MultipleLines'])
dataset['OnlineSecurity']=le.fit_transform(dataset['OnlineSecurity'])
dataset['OnlineBackup']=le.fit_transform(dataset['OnlineBackup'])
dataset['DeviceProtection']=le.fit_transform(dataset['DeviceProtection'])
dataset['TechSupport']=le.fit_transform(dataset['TechSupport'])
dataset['StreamingTV']=le.fit_transform(dataset['StreamingTV'])
dataset['StreamingMovies']=le.fit_transform(dataset['StreamingMovies'])
dataset


# In[118]:


dataset['TotalCharges']=dataset['TotalCharges'].replace(to_replace=' ',value=np.nan)
dataset.isnull().sum()


# In[119]:


dataset=dataset.dropna(axis=0).reset_index()
dataset


# In[120]:


dataset=dataset.drop(['index'],axis=1)
dataset


# In[121]:


ct=ColumnTransformer([('encoder',OneHotEncoder(),[7,14,16])],remainder='passthrough')
dataset=ct.fit_transform(dataset)
dataset=pd.DataFrame(dataset)
dataset


# In[122]:


x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values


# In[123]:


y=y.astype(int)


# In[124]:


y


# In[136]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=177013)


# In[126]:


print(x_train.shape,y_train.shape)


# In[142]:


from imblearn.over_sampling import SMOTE
smt=SMOTE(random_state=177013)


# In[143]:


x_train_smote,y_train_smote=smt.fit_resample(x_train,y_train)


# In[144]:


from collections import Counter
print("Before SMOTE", Counter(y_train))
print("After SMOTE", Counter(y_train_smote))


# In[132]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train_smote,y_train_smote)


# In[145]:


lr_pred=lr.predict(x_test)
lr_pred


# In[147]:


from sklearn.metrics import confusion_matrix,accuracy_score
cm_lr=confusion_matrix(y_test,lr_pred)
cm_lr


# In[148]:


acc_lr=accuracy_score(y_test,lr_pred)
acc_lr


# In[149]:


from sklearn.tree import DecisionTreeClassifier
dsc=DecisionTreeClassifier()
dsc.fit(x_train_smote,y_train_smote)


# In[150]:


dsc_pred=dsc.predict(x_test)
dsc_pred


# In[151]:


cm_dsc=confusion_matrix(y_test,dsc_pred)
cm_dsc


# In[152]:


acc_dsc=accuracy_score(y_test,dsc_pred)
acc_dsc


# In[153]:


from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=400)
rfc.fit(x_train_smote,y_train_smote)


# In[154]:


rfc_pred=rfc.predict(x_test)
rfc_pred


# In[155]:


cm_rfc=confusion_matrix(y_test,rfc_pred)
cm_rfc


# In[156]:


acc_rfc=accuracy_score(y_test,rfc_pred)
acc_rfc


# In[157]:


from xgboost import XGBClassifier
xgb=XGBClassifier()
xgb.fit(x_train_smote,y_train_smote)


# In[158]:


xgb_pred=xgb.predict(x_test)
xgb_pred


# In[159]:


cm_xgb=confusion_matrix(y_test,xgb_pred)
cm_xgb


# In[160]:


acc_xgb=accuracy_score(y_test,xgb_pred)
acc_xgb


# In[161]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train_smote=sc.fit_transform(x_train_smote)
x_test=sc.transform(x_test)


# In[162]:


from keras.models import Input,Model
from keras.layers import Dense,Dropout


# In[163]:


i=Input(shape=[26])


# In[164]:


layer1=Dense(units=20,activation='relu')(i)
layer2=Dense(units=14,activation='relu')(layer1)
layer3=Dense(units=8,activation='relu')(layer2)
out=Dense(units=1,activation='sigmoid')(layer3)


# In[165]:


ann=Model(inputs=i,outputs=out)


# In[166]:


ann.summary()


# In[178]:


from keras.utils.vis_utils import plot_model


# In[179]:


plot_model(ann, to_file='ann_model_plot.png', show_shapes=True, show_layer_names=True)


# In[ ]:





# In[168]:


ann.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# In[169]:


ann.fit(x_train_smote,y_train_smote,batch_size=32,epochs=30)


# In[170]:


metrics=pd.DataFrame(ann.history.history)


# In[171]:


metrics


# In[172]:


metrics[['loss']].plot()


# In[173]:


metrics[['accuracy']].plot()


# In[174]:


y_pred=ann.predict(x_test)
y_pred=(y_pred>0.5)
y_pred


# In[175]:


cm=confusion_matrix(y_test,y_pred)
cm


# In[176]:


score=accuracy_score(y_test,y_pred)
score


# In[177]:


print("Logistic Regression Accuracy=>{:.2%}".format(acc_lr))
print("Decision Tree Classifier Accuracy=>{:.2%}".format(acc_dsc))
print("Random Forest Classifeir Accuracy=>{:.2%}".format(acc_rfc))
print("XGB Classifeir Accuracy=>{:.2%}".format(acc_xgb))
print("Neural Network Accuracy=>{:.2%}".format(score))


# In[ ]:




