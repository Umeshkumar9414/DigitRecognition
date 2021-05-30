#!/usr/bin/env python
# coding: utf-8

# In[118]:


from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
get_ipython().magic(u'matplotlib inline')

digits =load_digits()


# In[119]:


# digits['data'][0]


# In[120]:


plt.figure(figsize=(10,4))
for index, (image,label) in enumerate(zip(digits.data[0:20],digits.target[0:20])):
    plt.subplot(1,20,index+1)
    plt.imshow(np.reshape(image,(8,8)))
    plt.title('%i \n' %label)


# In[121]:


x_train,x_test,y_train,y_test=train_test_split(digits.data,digits.target,test_size=0.23,random_state=2)


# In[122]:


from sklearn.linear_model import LogisticRegression
logRege=LogisticRegression(solver='liblinear')
logRege.fit(x_train,y_train)


# In[123]:


y_pred=logRege.predict(x_test)


# In[124]:


plt.figure(figsize=(10,4))
num=15
for index,(image,result) in enumerate(zip(x_test[0:num],y_pred[0:num])):
    plt.subplot(1,num,index+1)
    plt.imshow(np.reshape(image,(8,8)))
    plt.title('%i \n' %result)


# In[125]:


score=logRege.score(x_test,y_test)
print(score)


# In[126]:


cm=metrics.confusion_matrix(y_test,y_pred)
# print(cm)


# In[115]:


plt.figure(figsize=(9,9))
sns.heatmap(cm,annot=True,fmt=".0f",linewidths=-.5,square=True,cmap='Blues_r')
plt.ylabel("Actual Value")
plt.xlabel("predicted values")
plt.title("Score {0}".format(score),size=15)

