import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
import sklearn

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Loading data
print('Loading and cleaning data... ')
mbti = pd.read_csv('data/mbti_1.csv.gz')

# Cleaning text
def cleanText(text):
    text = BeautifulSoup(text, "lxml").text
    text = re.sub(r'\|\|\|', r' ', text) 
    text = re.sub(r'http\S+', r'<URL>', text)
    return text

mbti['clean_text'] = mbti['posts'].apply(cleanText)
mbti.drop('posts', axis=1, inplace=True)
mbti = mbti.sample(frac=1)

train, test = np.split(mbti, [int(.8*len(mbti))])

# Count vectorizer counts occurences of words for each entry.
count_vectorizer = CountVectorizer()
# Use this vector to classify. 
counts = count_vectorizer.fit_transform(train['clean_text'].values)

print('Training... ')
classifier = MultinomialNB()
targets = train['type'].values
classifier.fit(counts, targets)

# To test the model, we run the test dataset (not seen by model)
print('Testing model... ')
examples = list(test['clean_text'])
example_counts = count_vectorizer.transform(examples)
predictions = classifier.predict(example_counts)

example_vals = list(test['type'])

def confusion(predicted, real):
    """ will give a matrix containing the results"""
    confusion_matrix = np.zeros((len(targets), len(targets)))
    # Insert calculation... To come... 
    return confusion_matrix

num_correct = 0
for i in range(len(examples)):
    predicted = predictions[i]
    real = example_vals[i]
    if predicted == real:
        num_correct +=1
    #print(i)
    #print(f'predicted: {predicted}')
    #print(f'real: {real} \n')
print(f'Accuracy: {round((num_correct*100)/len(predictions),2)}%')



