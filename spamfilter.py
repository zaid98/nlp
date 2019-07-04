import nltk
from nltk.corpus import stopwords
import string
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

messages = pd.read_csv('spam.csv', encoding='latin-1')
messages.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1,inplace=True)
messages = messages.rename(columns={'v1': 'class','v2': 'text'})

def process_text(text):
   
    np = [char for char in text if char not in string.punctuation]
    np = ''.join(np)
    
    
    cleaned = [word for word in np.split() if word.lower() not in stopwords.words('english')]
    
    
    return cleaned

mail_train, mail_test, class_train, class_test = train_test_split(messages['text'],messages['class'],test_size=0.2)

pipeline = Pipeline([
    ('count',CountVectorizer(analyzer=process_text)), 
    ('tfidf',TfidfTransformer()), 
    ('classifier',MultinomialNB())
])

pipeline.fit(mail_train,class_train)
predictions = pipeline.predict(mail_test)
