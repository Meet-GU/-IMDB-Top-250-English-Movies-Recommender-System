#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set pandas display option
pd.set_option('display.max_columns', None)

# Load the dataset
df = pd.read_csv("IMDB_Top250Engmovies2_OMDB_Detailed.csv")
print(df)

# Preprocess the 'Plot' column
df['clean_plot'] = df['Plot'].str.lower()
df['clean_plot'] = df['clean_plot'].apply(lambda x: re.sub('[^a-zA-Z]', ' ', x))
df['clean_plot'] = df['clean_plot'].apply(lambda x: re.sub('\s+', ' ', x))

# Tokenize the 'clean_plot' column
nltk.download('punkt')
df['clean_plot'] = df['clean_plot'].apply(lambda x: nltk.word_tokenize(x))

# Remove stopwords and filter tokens
nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words('english')

def remove_stopwords(sentence):
    return [word for word in sentence if word not in stop_words and len(word) >= 3]

df['clean_plot'] = df['clean_plot'].apply(remove_stopwords)

# Process 'Genre', 'Actors', and 'Director' columns
df['Genre'] = df['Genre'].apply(lambda x: x.split(','))
df['Actors'] = df['Actors'].apply(lambda x: x.split(',')[:4])
df['Director'] = df['Director'].apply(lambda x: x.split(','))

def clean_column(data):
    return [word.lower().replace(' ', '') for word in data]

df['Genre'] = df['Genre'].apply(clean_column)
df['Actors'] = df['Actors'].apply(clean_column)
df['Director'] = df['Director'].apply(clean_column)

# Combine all columns into a single input column
columns = ['clean_plot', 'Genre', 'Actors', 'Director']

def combine_columns(row):
    combined = ''
    for col in columns:
        combined += ''.join(row[col]) + ' '
    return combined.strip()

df['clean_input'] = df.apply(combine_columns, axis=1)
df = df[['Title', 'clean_input']]

# Feature extraction using TF-IDF
tfidf = TfidfVectorizer()
features = tfidf.fit_transform(df['clean_input'])

# Compute cosine similarity
cosine_sim = cosine_similarity(features, features)

# Create an index for the titles
index = pd.Series(df['Title'])

# Recommendation function
def recommend_movies(title):
    try:
        movies = []
        idx = index[index == title].index[0]
        score = pd.Series(cosine_sim[idx]).sort_values(ascending=False)
        top10 = list(score.iloc[1:11].index)
        for i in top10:
            movies.append(df['Title'][i])
        return movies
    except IndexError:
        return ["Movie not found in the dataset"]

# Example usage
print(recommend_movies('The Dark Knight Rises'))
print(recommend_movies('The Shawshank Redemption'))
print(recommend_movies('Spider-Man: Homecoming'))
