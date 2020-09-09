# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 02:02:15 2020

@author: Sujay J
"""

import pandas as pd
messages = pd.read_csv("spam Classifier\SMSSpamCollection",sep = '\t',names = ["labels","message"])
import nltk
import re #regular Expressions
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

ps = PorterStemmer()
wordnet  = WordNetLemmatizer()
corpus = []

for i in range(len(messages)):
    review = re.sub('[^a-zA-Z]',' ' , messages['message'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word)for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000)
X = cv.fit_transform(corpus).toarray()

y = pd.get_dummies(messages['labels'])
y = y.iloc[:,1].values


from sklearn.model_selection import train_test_split
X_train ,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.naive_bayes import MultinomialNB

model=MultinomialNB().fit(X_train,y_train)

y_pred=model.predict(X_test)

y_pred

from sklearn.metrics import confusion_matrix,accuracy_score

confu = confusion_matrix(y_test,y_pred)

accu = accuracy_score(y_test,y_pred)