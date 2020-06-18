#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install joblib


# In[1]:


import pandas as pd
import joblib
df = pd.read_csv('train.csv')


# In[4]:


comment = df['comment_text']
comment = comment.as_matrix()
label = df[['toxic', 'severe_toxic' , 'obscene' , 'threat' , 'insult' , 'identity_hate']]
label = label.as_matrix()
df['comment_text'].fillna("unknown", inplace=True)


# In[5]:


comments = []
labels = []

for i in range(comment.shape[0]):
    if len(comment[i])<=400:
        comments.append(comment[i])
        labels.append(label[i])
len(comments)        


# In[6]:


from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
    


# In[7]:


corpus = []
ps = PorterStemmer()
for j in comments:
    j = re.sub('[^a-zA-Z]', ' ', j)
    j = j.lower()
    j = j.split()
    j = [word for word in j if not word in set(stopwords.words('english'))]
    j = [ps.stem(word) for word in j]
    j = " ".join(j)
    corpus.append(j)


# In[8]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 2500)


# In[9]:


features = cv.fit_transform(corpus).toarray()


# In[12]:


import numpy as np
labels = np.asarray(labels)


# In[13]:


from skmultilearn.problem_transform import BinaryRelevance
from sklearn.svm import SVC
classifier = BinaryRelevance(classifier = SVC(gamma = 'auto'), require_dense = [False, True])
classifier.fit(features, labels)


# In[14]:


joblib.dump(classifier, "joblib_prmodel")


# In[15]:


joblib.load("joblib_prmodel")


# In[16]:


df1 = pd.read_csv('test.csv')


# In[19]:


com = df1['comment_text']
com = com.as_matrix()


# In[47]:


corpus1 = []
com[48] = re.sub('[^a-zA-Z]', ' ', com[48])
com[48] = com[48].lower()
com[48] = com[48].split()
com[48] = [word for word in com[48] if not word in set(stopwords.words('english'))]
com[48] = [ps.stem(word) for word in com[48]]
com[48] = " ".join(com[48])
corpus1.append(com[48])


# In[48]:


test_features = cv.transform(corpus1).toarray()

