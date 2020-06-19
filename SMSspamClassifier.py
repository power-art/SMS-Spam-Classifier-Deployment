#!/usr/bin/env python
# coding: utf-8

# In[3]:


#importing the dataset
import pandas as pd
message = pd.read_csv("SMSSpamCollection",sep='\t',names=["label","message"])


# #### Data cleaning and preprocessing

# In[4]:


import re  #this library is used for regular expression
import nltk 


# In[5]:


from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# In[6]:


ps = PorterStemmer()
corpus = []


# In[7]:


for i in range(0,len(message)):
    #i will remove all the character except a to z or
    #A to Z and replace it with blank 
    review = re.sub('[^a-zA-Z]',' ',message['message'][i])
    review = review.lower()
    review = review.split()
    #stemming
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)


# In[8]:


#creating the bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2500)
x = cv.fit_transform(corpus).toarray()


# In[9]:


#Extractiing dependent variable from the dataset
y = pd.get_dummies(message['label'])


# In[10]:


y = y.iloc[:,1].values


# In[11]:


#creating a pickle file for the count vectorizer
import pickle
pickle.dump(cv,open('cv-transform.pkl','wb'))


# In[12]:


#train test split
from sklearn.model_selection import train_test_split
X_train, X_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)


# In[13]:


#it does work very well for nlp problem
from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train,y_train)


# In[15]:


# Creating a pickle file for the Multinomial Naive Bayes model
filename = 'SMS-sPAM-model.pkl'
pickle.dump(spam_detect_model, open(filename, 'wb'))


# In[ ]:





# In[16]:


y_pred = spam_detect_model.predict(X_test)


# In[17]:


from sklearn.metrics import confusion_matrix
confusion_m = confusion_matrix(y_test,y_pred)


# In[18]:


confusion_m


# In[19]:


from sklearn.metrics import accuracy_score
score = accuracy_score(y_test,y_pred)
score


# In[ ]:




