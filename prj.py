#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("train.csv")


# In[3]:


comment = df['comment_text'].values


# In[4]:


label = df[['toxic', 'severe_toxic' , 'obscene' , 'threat' , 'insult' , 'identity_hate']].values


# In[5]:




# In[6]:


comments = []
labels = []

for ix in range(comment.shape[0]):
    if len(comment[ix])<=200:
        comments.append(comment[ix])
        labels.append(label[ix])


# In[7]:


labels = np.asarray(labels)


# In[8]:


import string
print(string.punctuation)
punctuation_edit = string.punctuation.replace('\'','') +"0123456789"
print (punctuation_edit)
outtab = "                                         "
trantab = str.maketrans(punctuation_edit, outtab)


# In[9]:




# In[10]:


import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer


# In[11]:


lemmatiser = WordNetLemmatizer()
stemmer = PorterStemmer()
nltk.download('wordnet')


# In[12]:


for i in range(len(comments)):
    comments[i] = comments[i].lower().translate(trantab)
    comments[i] = comments[i].split()
    comments[i] = [word for word in comments[i] if not word in set(stopwords.words('english')) ]
    l = []
    for word in comments[i]:
        l.append(stemmer.stem(lemmatiser.lemmatize(word,pos="v")))
    comments[i] = " ".join(l)


# In[13]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.95, min_df=5)
features = vectorizer.fit_transform(comments).toarray()


# In[14]:


features.shape


# In[15]:


from skmultilearn.problem_transform import BinaryRelevance
from sklearn.svm import SVC
classifier = BinaryRelevance(classifier = SVC(kernel='linear', C=1,probability=True), require_dense = [False, True])
classifier.fit(features, labels)


# In[31]:


import joblib


# In[32]:


joblib.dump(vectorizer, "job_tfv")


# In[56]:


joblib.dump(classifier, "job_model")


# In[58]:


com = input("Enter your comment:")


# In[59]:


com = com.lower().translate(trantab)


# In[101]:



# In[60]:


x = []
com = com.split()
for word in com:    
    x.append(stemmer.stem(lemmatiser.lemmatize(word,pos="v")))
com = " ".join(x)


# In[61]:


cc = []
cc.append(com)


# In[62]:


test = vectorizer.transform(cc).toarray()


# In[63]:


pred = classifier.predict(test)


# In[51]:
if pred[:,:].toarray().any() == 1:
    print("abusive comment")
else:
    print("comment is fine")
classes = ['toxic', 'severe_toxic' , 'obscene' , 'threat' , 'insult' , 'identity_hate']
for k in range(0,6):
    if pred[0,k] == 1:
        print(classes[k])



# In[64]:




# In[30]:



# In[ ]:




