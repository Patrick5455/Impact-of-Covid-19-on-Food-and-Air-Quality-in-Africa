import tweepy
import os
import sys
import datetime as dt
import pandas as pd
import numpy as np
from tweepy import OAuthHandler
from tweepy import API
from tweepy import Cursor
from nltk.corpus import stopwords
import preprocessor as p


# Variables that contains the user credentials to access Twitter API
consumer_key = os.environ.get('TWITTER_API_KEY')
consumer_secret = os.environ.get('TWITTER_API_SECRET')
access_token = os.environ.get('TWITTER_ACCESS_TOKEN')
access_token_secret = os.environ.get('TWITTER_ACCESS_TOKEN_SECRET')

# This handles Twitter authetification and the connection to Twitter Streaming API
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = API(auth, wait_on_rate_limit=True)

keywords = "airquality"
startDate = "2020-03-3"
tweets = Cursor(api.search, q=keywords, lang="en", since=startDate).items(2)

for tweet in tweets:
    print(tweet.text)
