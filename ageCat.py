import numpy as np
import pandas as pd

import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
    
# datasets used for demographic categorization
hm_data = pd.read_csv('Datasets/cleaned_hm.csv')
demo_data = pd.read_csv('Datasets/demographic.csv')

df = pd.merge(hm_data, demo_data, on='wid') # connect demographic and text data
age_data = df[['cleaned_hm', 'age']] # dataframe containing cleaned text and age column
gender_data = df[['cleaned_hm', 'gender']] # dataframe containing cleaned text and gender column

# clean up age column to not have any NaN/strings
age = age_data['age']
age = pd.to_numeric(age, errors='coerce')
age_data['age'] = age
age_data = age_data[age_data['age'].notna()]
age_data['age'] = age_data['age'].round(0).astype(int) # make all age rounded to the nearest integer instead of 20.0 --> 20


age_data['30'] = np.where(age_data['age'] <= 30, 0, 1) # split dataset into younger than 30 or not, value counts = 0:53773, 1:46621

age_data['cleaned_hm'] = age_data['cleaned_hm'].astype(str).str.lower()

print(age_data['cleaned_hm'].head(5))

vectorizer = CountVectorizer()
features_train_hm = vectorizer.fit_transform(age_data['cleaned_hm'])
train_array_hm = features_train_hm.toarray()

X_train, X_test, y_train, y_test = train_test_split(train_array_hm, age_data['30'], test_size = 0.3, random_state = 42)

lr = LogisticRegression()
lr = lr.fit(X_train, y_train)
print(lr.score(X_test, y_test))





