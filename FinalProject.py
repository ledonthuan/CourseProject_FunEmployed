"""
*****************************
1. use a pretrained sentiment analysis model to apply to google documents

2. test the model - how to test????
3. take the top 2000 documents from google and apply sentiment analysis
4. probabilistic topic modeling to see what topics 
    - topics: how to isolate culture or age??
        - use the happydb dataset, categorize by age groups and then classify
        - Naive Bayes classifier

5. return the the user the percentage for each topic (i.e. culture or age) from
both positive and negative sentiments

note: eliminate documents/URLs that we are not able to find topic
*****************************
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer # vader = pretrained sentiment analysis model
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist

import requests
from bs4 import BeautifulSoup
from googleapiclient.discovery import build

def main():
    # Build a service object for interacting with the API. Visit
    # the Google APIs Console <http://code.google.com/apis/console>
    # to get an API key for your own application.
    service = build(
        "customsearch", "v1", developerKey="AIzaSyCDJuoOWM1RgNbBBYCwEMVhkSakngokE_8"
    )
    uri = []
    for page in range(10): # 10 for 100
        res = (
            service.cse()
            .list(
                q="peoples opnion about their father",
                cx="573388c6cd38f4a11",
                num = 10,  # integer between 1 and 10
                start = (page*10)+1, # need to make paginated calls to return more - for loop, count starts at 1, not 0
            )
            .execute()
        )
        for i in range(len(res['items'])):
            uri.append(res['items'][i]['link'])
    print("URI has {} saved links".format(len(uri)))

    text_data = []
    text_size = 0
    for link in uri:
        print(link)

        try:
            f = requests.get(link)
            #soup = BeautifulSoup(f.text, features='lxml').get_text()
            soup = BeautifulSoup(f.text, features='lxml')
            mainSoup = soup.find('main').text
            mainSoup = mainSoup.replace('\n', ' ')
            # print(mainSoup)
            text_size += 1
            text_data += [mainSoup]
        except Exception:
            print("URL error, skipped")
            pass
    
    #print(text_data[0:5])
    print("text list has {} entries".format(len(text_data)))


    df = pd.DataFrame(text_data, columns=['text'])
    # this dataframe will contain all the text data collected from the google query
    # df = pd.dataframe(*list*, columns = ['text'])

    # bring text in as dataframe with columns: designator, text
    df['text'] = df['text'].astype(str).str.lower()
    regexp = RegexpTokenizer('\w+')
    df['text_token'] = df['text'].apply(regexp.tokenize)

    # remove stopwords
    stopwords = nltk.corpus.stopwords.words("english")
    df['text_token'] = df['text_token'].apply(lambda x: [item for item in x if item not in stopwords])

    df['text_string'] = df['text_token'].apply(lambda x:' '.join([item for item in x if len(item)>2]))
    all_words = ' '.join([word for word in df['text_string']])
    tokenized_words = nltk.tokenize.word_tokenize(all_words)

    fdist = FreqDist(tokenized_words)
    df['text_string_fdist'] = df['text_token'].apply(lambda x: ' '.join([item for item in x if fdist[item] >= 1]))
    # improvement: consider TF-IDF weighting more heavily


    #lemmatization
    wordnet_lem = WordNetLemmatizer()
    df['text_string_lem'] = df['text_string_fdist'].apply(wordnet_lem.lemmatize)

    #sentiment analysis using vader
    sat = SentimentIntensityAnalyzer()
    df['polarity'] = df['text_string_lem'].apply(lambda x: sat.polarity_scores(x))
    df = pd.concat([df['polarity'].apply(pd.Series)], axis=1)
    df['sentiment'] = df['compound'].apply(lambda x: 'positive' if x>0 else 'neutral' if x==0 else 'negative') # https://www.kirenz.com/post/2021-12-11-text-mining-and-sentiment-analysis-with-nltk-and-pandas-in-python/text-mining-and-sentiment-analysis-with-nltk-and-pandas-in-python/

    print(df['sentiment'].value_counts())
    print(df['compound'])



    """
    If I wanted to download the dataset directly using the KaggleAPI
    - i chose to download the dataset directly to my local environment
    os.environ['KAGGLE_USERNAME'] = 'donnythedino'
    os.environ['KAGGLE_KEY'] = '40c0f5bad56fb689cca62c774a7a5c31'

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()

    api.dataset_list_files('datasets/ritresearch/happydb').files()
    api.dataset_download_files('datasets/ritresearch/happydb', path='.')
    """

if __name__ == "__main__":
    main()




