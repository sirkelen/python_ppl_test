import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
import sklearn

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Cleaning text
def cleanText(text):
    text = BeautifulSoup(text, "lxml").text
    text = re.sub(r'\|\|\|', r' ', text) 
    text = re.sub(r'http\S+', r'<URL>', text)
    return text

def results(category, predicted):
    num_correct = 0
    for i, val in enumerate(test[category].values):
        predict = predicted[i]
        real = val
        if predict == val:
            num_correct +=1
        #print(i)
        #print(f'predicted: {predicted}')
        #print(f'real: {real} \n')
    print(f'Accuracy Native Bayes {category}: {round((num_correct*100)/len(predicted),2)}%')

# Loading data
print('Loading and cleaning data... ')
mbti = pd.read_csv('data/mbti_1.csv.gz')

# Preparing samples:
mbti['clean_text'] = mbti['posts'].apply(cleanText)
# Creating individual columns for EI/NS/FT/PJ:
for i, cat in enumerate(['ei', 'ns', 'ft', 'pj']):
    mbti[cat] = mbti['type'].apply(lambda x: x[i])
# No need to keep the column of original text...
mbti.drop('posts', axis=1, inplace=True)
# Scrambling the data so the order is random.
mbti = mbti.sample(frac=1)

train, test = np.split(mbti, [int(.8*len(mbti))])


# Building models with pipeline...
pipeline = Pipeline([
    ('vectorizer',  CountVectorizer(ngram_range=(1,  2))),
    ('classifier',  MultinomialNB()) ])

ei_pipeline = pipeline
ns_pipeline = pipeline
ft_pipeline = pipeline
pj_pipeline = pipeline

print('Training E/I...')
ei_pipeline.fit(train['clean_text'].values, train['ei'].values)
print('Predicting E/I...')
ei_predicted = ei_pipeline.predict(test['clean_text'].values)
results('ei', ei_predicted)

print('Training N/S...')
ns_pipeline.fit(train['clean_text'].values, train['ns'].values)
print('Predicting N/S...')
ns_predicted = ns_pipeline.predict(test['clean_text'].values)
results('ns', ns_predicted)

print('Training F/T...')
ft_pipeline.fit(train['clean_text'].values, train['ft'].values)
print('Predicting F/T...')
ft_predicted = ft_pipeline.predict(test['clean_text'].values)
results('ft', ft_predicted)

print('Training P/J...')
pj_pipeline.fit(train['clean_text'].values, train['pj'].values)
print('Predicting P/J...')
pj_predicted = pj_pipeline.predict(test['clean_text'].values)
results('pj', pj_predicted)

def confusion(predicted, real):
    """ will give a matrix containing the results"""
    confusion_matrix = np.zeros((len(targets), len(targets)))
    # Insert calculation... To come... 
    return confusion_matrix