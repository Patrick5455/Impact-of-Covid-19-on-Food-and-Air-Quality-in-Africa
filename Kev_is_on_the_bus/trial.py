import pandas as pd
import tweepy
from tweepy import OAuthHandler
from tweepy import API
from tweepy import Cursor
from datetime import datetime
from collections import Counter
import os
import sys
import csv


API_KEY = os.environ.get('TWITTER_API_KEY')
API_SECRET_KEY = os.environ.get('TWITTER_API_SECRET')
ACCESS_TOKEN = os.environ.get('TWITTER_ACCESS_TOKEN')
ACCESS_TOKEN_SECRET = os.environ.get('TWITTER_ACCESS_TOKEN_SECRET')

auth = OAuthHandler(API_KEY, API_SECRET_KEY)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth, wait_on_rate_limit=True)
auth_api = API(auth)

search_words = "#airpolution"
date_since = "2018-04-1"
geocode = "-1.286389,36.817223,40km"
# Collect tweets
tweets = tweepy.Cursor(api.search, q="*", geocode=geocode, lang="en",
                       since=date_since).items(2)
# Iterate and print tweets
for tweet in tweets:
    tweet = tweet._json
    print(tweet['text'])
