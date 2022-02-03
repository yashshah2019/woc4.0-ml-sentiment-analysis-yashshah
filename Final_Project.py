#!/usr/bin/env python
# coding: utf-8

# In[321]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
import nltk
import regex
import random


# In[322]:


from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


# In[323]:


df = pd.read_csv('C:\\Users\\YASH\\Downloads\\judge-1377884607_tweet_product_company.csv',encoding = 'latin1')
df


# In[324]:


df.shape


# In[325]:


df.head()


# In[326]:


df.dtypes


# In[327]:


df.describe()


# In[328]:


df.isnull().sum()


# In[329]:


df.drop(axis = 1,columns = 'emotion_in_tweet_is_directed_at',inplace = True)


# In[330]:


df['tweet_text'].fillna('neutral comment',inplace = True)


# In[331]:


df.isnull().sum()


# In[332]:


temp = df['is_there_an_emotion_directed_at_a_brand_or_product'].value_counts()
temp


# In[333]:


df = df.loc[df['is_there_an_emotion_directed_at_a_brand_or_product'] != 'No emotion toward brand or product',:]


# In[334]:


temp = df['is_there_an_emotion_directed_at_a_brand_or_product'].value_counts()
temp


# In[335]:


plt.bar(temp.index,temp,width = 0.4)
plt.title("Analysis of Products")
plt.xlabel("Products")
plt.ylabel("Count")


# In[336]:


from sklearn.preprocessing import LabelEncoder


# In[337]:


labelencoder = LabelEncoder()
df['is_there_an_emotion_directed_at_a_brand_or_product_int'] = labelencoder.fit_transform(df['is_there_an_emotion_directed_at_a_brand_or_product'])
# df['emotion_in_tweet_is_directed_at_int'] = labelencoder.fit_transform(df['emotion_in_tweet_is_directed_at'])


# In[338]:


df = df.replace('[^a-zA-Z]', ' ', regex = True)


# In[339]:


df = df.replace('https* | www*',' ',regex = True)


# In[340]:


df = df.replace('\s+',' ',regex = True)


# In[341]:


# Lower all the text.
df['tweet_text'] = df['tweet_text'].str.lower()
df['is_there_an_emotion_directed_at_a_brand_or_product'] = df['is_there_an_emotion_directed_at_a_brand_or_product'].str.lower()


# In[342]:


df = df.loc[df['is_there_an_emotion_directed_at_a_brand_or_product'] != 'i can t tell',:]


# In[343]:


df['tweet_text'][0]


# In[344]:


nltk.download('stopwords')


# In[345]:


ls = WordNetLemmatizer()


# In[346]:


for i in df['tweet_text']:
    tweets = nltk.word_tokenize(i)
    tweets = [ls.lemmatize(word) for word in tweets if not word in set(stopwords.words('english'))]
    tweets = ' '.join(tweets)
    i = tweets


# In[347]:


df.get(5)


# In[348]:


df


# In[349]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000)
X = cv.fit_transform(df['tweet_text']).toarray()


# In[350]:


X.shape


# In[351]:


cv.get_feature_names()[:15]


# In[352]:


y = df['is_there_an_emotion_directed_at_a_brand_or_product_int']


# In[353]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)


# In[354]:


bog_represent = pd.DataFrame(X_train,columns = cv.get_feature_names())


# In[355]:


bog_represent


# In[ ]:


# Implemented 3 Algorithms for classifying tweet as positive or negative
# 1. MultinomialNB Algorithm
# 2. Passive Aggressive Classifier Algorithm
# 3. Decision Tree Algorithm

# For each and every algorithm score and confusion matrix is calculated


# In[356]:


# MultinomialNB Algorithm


# In[357]:


from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
mnb = MultinomialNB()


# In[358]:


mnb.fit(X_train,y_train)
y_predicted = mnb.predict(X_test)
score = metrics.accuracy_score(y_test,y_predicted)
score*100


# In[359]:


confusion_matrix = metrics.confusion_matrix(y_test,y_predicted)
confusion_matrix
# 1st col = Negative emotion
# 2nd col = Positive emotion


# In[360]:


# Passive Aggressive Classifier Algorithm


# In[361]:


from sklearn.linear_model import PassiveAggressiveClassifier
pac = PassiveAggressiveClassifier()


# In[362]:


pac.fit(X_train,y_train)
y_predicted = pac.predict(X_test)
score = metrics.accuracy_score(y_test,y_predicted)
score*100


# In[363]:


confusion_matrix = metrics.confusion_matrix(y_test,y_predicted)
confusion_matrix
# 1st col = Negative emotion
# 2nd col = Positive emotion


# In[364]:


# Decision Tree Algorithm


# In[367]:


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train,y_train)


# In[368]:


dtc.fit(X_train,y_train)
y_predicted = dtc.predict(X_test)
score = metrics.accuracy_score(y_test,y_predicted)
score*100


# In[369]:


confusion_matrix = metrics.confusion_matrix(y_test,y_predicted)
confusion_matrix
# 1st col = Negative emotion
# 2nd col = Positive emotion

