import tweepy
import os
import sys
import datetime as dt
import pandas as pd
import numpy as np
import re
import json
from tweepy import OAuthHandler
from tweepy import API
from tweepy import Cursor
from nltk.corpus import stopwords
import preprocessor as p


# Variables that contains the user credentials to access Twitter API
consumer_key = ''
consumer_secret = ''
access_token = ''
access_token_secret =''

# This handles Twitter authetification and the connection to Twitter Streaming API
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = API(auth, wait_on_rate_limit=True)

# keywords = "pollution"
startDate = "2020-03-3"
# tweets = Cursor(api.search, q=keywords, lang="en", since=startDate).items(2)

# for tweet in tweets:
#     print(tweet.text)
keywords=['covid','pollution']
searchCountry=["Kenya"]
geoCode="-33.918861, 18.423300, 40km "

places = api.geo_search(query=searchCountry, granularity="country")
place_id = places[0].id


def gettweets(keyword):
    data=[] #empty list to which tweet details will be added
    counter=0 #couter to keep track of every iteration
    for tweet in tweepy.Cursor(api.search, q='\"{}\" -filter:retweets'.format(keyword) and ("place:%s" % place_id),  since=startDate,count=5, lang='en', tweet_mode='extended').items():
        tweet_details = {}
        tweet_details['name'] = tweet.user.screen_name
        tweet_details['tweet'] = tweet.full_text
        # tweet_details['retweets'] = tweet.retweet_count
        tweet_details['location'] = tweet.user.location
        tweet_details['created'] = tweet.created_at.strftime("%d-%b-%Y")
        tweet_details['followers'] = tweet.user.followers_count
        # tweet_details['is_user_verified'] = tweet.user.verified

        data.append(tweet_details)
        counter +=1
        if counter==5:
            break
        else:
            pass
        with open('data/{}.json'.format(keyword), 'w') as f: json.dump(data, f)
    print('done!')

if __name__ == "__main__":
    print('Starting to stream...')
    for keyword in keywords:
        gettweets(keyword)
    print('finished!')   

#helper function for removing odd characters from the tweets
def clean_tweet(tweet):
    return ' '.join(re.sub('(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)', ' ', tweet).split())

#create individual datframes for each file
def create_covid_df():
    covid_df=pd.read_json('data/covid.json', orient='records')
    covid_df['clean_tweet']=covid_df['tweet'].apply(lambda x: clean_tweet(x))
    covid_df['keyword']='covid'
    covid_df.drop_duplicates(subset=['name'],keep='first', inplace =True)
    return covid_df

def create_pollution_df():
    pollution_df=pd.read_json('data/pollution.json', orient='records')
    pollution_df['clean_tweet']=pollution_df['tweet'].apply(lambda x: clean_tweet(x))
    pollution_df['keyword']='pollution'
    pollution_df.drop_duplicates(subset=['name'],keep='first', inplace =True)
    return pollution_df

#join the dataframes
def join_dfs():
    covid_df = create_covid_df()
    pollution_df = create_pollution_df()
    frames = [covid_df, pollution_df]
    keyword_df = pd.concat(frames, ignore_index=True)
    return keyword_df

k=join_dfs()
print(k)   

