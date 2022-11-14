from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd 
import time 
import re

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
english_stop_words = stopwords.words('english')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer


#preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# machine learning

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import classification_report

pd.set_option('display.max_columns',None)

headers=['Tweet_ID','Entity','Sentiment','Tweet_content']

train_df=pd.read_csv('archive/twitter_training.csv', sep=',', names=headers)

valid_df=pd.read_csv('archive/twitter_validation.csv', sep=',', names=headers)

train_df= train_df.drop_duplicates()

# encoder for target feature
from sklearn import preprocessing
lb = preprocessing.LabelEncoder()
train_df['Sentiment']=lb.fit_transform(train_df['Sentiment'])

train_df.dropna(axis=0, inplace=True)

tweet_train  = train_df["Tweet_content"]
tweet_valid=valid_df["Tweet_content"]
target=train_df['Sentiment']

REPLACE_WITH_SPACE = re.compile("(@)")
SPACE = " "

def preprocess_reviews(reviews):  
    reviews = [REPLACE_WITH_SPACE.sub(SPACE, line.lower()) for line in reviews]
    
    return reviews

reviews_train_clean = preprocess_reviews(tweet_train)
reviews_valid_clean = preprocess_reviews(tweet_valid)

def remove_stop_words(corpus):
    removed_stop_words = []
    for review in corpus:
        removed_stop_words.append(
            ' '.join([word for word in review.split()  if word not in english_stop_words]))
    return removed_stop_words

no_stop_words_train = remove_stop_words(reviews_train_clean)
no_stop_words_valid = remove_stop_words(reviews_valid_clean)

def get_stemmed_text(corpus):
    stemmer = PorterStemmer()

    return [' '.join([stemmer.stem(word) for word in review.split()]) for review in corpus]


stemmed_reviews_train = get_stemmed_text(no_stop_words_train)
stemmed_reviews_test = get_stemmed_text(no_stop_words_valid)

tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(stemmed_reviews_train)
X = tfidf_vectorizer.transform(stemmed_reviews_train)
X_test = tfidf_vectorizer.transform(stemmed_reviews_test)

X_train, X_val, y_train, y_val = train_test_split(X, target, train_size=0.75)

# RandomForestClassifier
text_classifier = RandomForestClassifier(n_estimators=500, random_state=0)
text_classifier.fit(X_train, y_train)

y_pred=text_classifier.predict(X_val)
print(classification_report(y_val,y_pred))

# KNeighborsClassifier
text_classifier2 = KNeighborsClassifier(
    n_neighbors=273)
text_classifier2.fit(X_train, y_train)
y_pred2 = text_classifier2.predict(X_val)
print(classification_report(y_val, y_pred2))

# DecisionTreeClassifier
text_classifier3 = DecisionTreeClassifier(criterion="gini")
text_classifier3.fit(X_train, y_train)
y_pred3 = text_classifier3.predict(X_val)
print(classification_report(y_val, y_pred3))


# TODO: add dictionary for data cleaning to eliminate jargon
