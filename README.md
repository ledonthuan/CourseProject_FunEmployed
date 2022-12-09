OVERVIEW:
link to video demo: https://youtu.be/0MU47rbcGjM
Overall, this project is incomplete and in 2 separate pieces currently. The goal of the project was to create an intelligent browser addition to a users google search. The general process would be to take a user's query, perform sentiment analysis across the different URLs that were returned and to create a model to categortize the text from the URLs to estimate the age of the opinion holder (in this case I created a column of the dataset that separated the data between those that were 30 years of age and younger and those that were older). By labeling the main text from the user's query, the user would be able to see who generally holds a positive opinion in regards to their query or who generally holds a negative opinion. If you're wondering why I sound like im slightly dying it's cause the flu has had me knocked for 2 weeks :(

Since this is an unfinished project there are many improvements to be made in the future:
    - connect the opinion analyzer of the URLs to the age categorizer
    - fully apply the python code to chrome extension by potentially compiling with Rapydscript
    - consider TF-IDF weighting more heavily when cleaning the text
    - cleaning the URL main text more heavily because each URL is structured differently making it difficult to create a completely clean dataset

TEAM MEMBER CONTRIBUTIONS:
This was a solo project so all research and code was put together by Don-Thuan Le. The project was originally planned to be implemented as a chrome extension because I thought that would be the easiest to access the Google Search results - so initally the project was geared towards creating this extension. Then having come upon the Google Search API, I pivoted as to just a python script to later be compiled into a chrome extension if possible. Although my 20 hours of code is not as substantial as many other projects, it gave me a lot of time to research and learn about text mining, analytics, APIs, and most importantly attempting to implement something that I could see to be very useful...if it worked. 

RELATED WORK, LIBRARIES, MODELS:
I based my project off the HappyDB text database that was provided on Kaggle and took inspiration from Chen Chen(https://www.kaggle.com/code/powderist/happydb-analysis/notebook) who did analysis based on gender. I wanted to do attempt age because I believe it is easier to label vocabulary based on the age of an individual rather than the way the write.
Libraries and Models that I used included:
    - nltk: for a pretrained sentiment analysis analyzer (vader) and text cleaning (tokenzation, stopword removal, etc.)
    - numpy
    - pandas
    - sklearn: split dataset and implement LogisticRegression\
    - bs4: Beautifulsoup to help clean the data from the URLs
Dataset:
    - HappyDB Kaggle
APIs:
    - Google Search API

CODE STRUCTURE:
This project consists of two main code files:
 1. FinalProject.py: implements the Google Search API and the cleaning of the URL text to be inserted into the vader sentiment analyzer
 2. ageCat.py: Reads in the cleaned_hm and demographic dataset from HappyDB to be split into a dataset of text and age of those younger than or equal to 30 vs those older than 30
The goal of the project was to link the model created by the ageCat.py with the URL text data that was collected but the implementation of the logistic Regression was not successful.

SET UP AND RUN CODE:
Different from the class, I updated my from 2.7 to 3.5 to be able to run the nltk library.
For the FinalProject.py, it is currently using my personal key for the Google Search API which has a limit of 100 requests per day (you may want to get your own key to insert into the developer key field for more queries). The Google Search API functions by pulling the URLs page by page allowing one to get 10 URLs per a request. You can change the number of loops on this request to collect more text from returned URLs.
In this github you will also find both cleaned_hm.csv and demographic.csv which you can download to your local machine or pull from kaggle. These were the files read in by ageCat.py.

