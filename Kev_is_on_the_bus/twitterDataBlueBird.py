import pandas as pd
import numpy as np
import os
import re
import logging
import nltk
nltk.download('stopwords')
nltk.download('punkt')
import string
from datetime import datetime, date, time, timedelta
import tweepy
from tweepy import OAuthHandler, API, Cursor, Stream
from tweepy.streaming import StreamListener
from nltk.corpus import stopwords
from textblob import TextBlob
from collections import Counter
import sys
import preprocessor as p
import GetOldTweets3 as got
from GetOldTweets3.manager import TweetCriteria
from bluebird import BlueBird


class dataAcq():
    def __init__(self, cols=None, auth=None):
        if cols is not None:
            self.cols = cols
        else:
            self.cols = ['id', 'created_at', 'original_text', 'clean_text',
                         'polarity', 'subjectivity', 'favorite_count', 'retweet_count',
                         'possibly_sensitive', 'hashtags', 'user_mentions', 'place']

    def cleanTweets(self, twitterText):
        print("Cleaning Tweets :) \n\n")
        tweet = p.clean(twitterText)

        # Happy Emoticons
        emoticonsHappy = set([
            ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
            ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
            '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
            'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
            '<3'])

        # Sad Emoticons
        emoticonsSad = set([
            ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
            ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
            ':c', ':{', '>:\\', ';('])

        # Emoji patterns
        emojiPattern = re.compile("[" u"\U0001F600-\U0001F64F"  # emoticons
                                      u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                      u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                      u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                      u"\U00002702-\U000027B0"
                                      u"\U000024C2-\U0001F251" "]+", flags=re.UNICODE)

        # Combine the sad and happy emoticons
        emoticons = emoticonsHappy.union(emoticonsSad)

        # tokenize the words in the tweet and get the stop words
        stopWords = set(stopwords.words('english'))
        wordTokens = nltk.word_tokenize(tweet)

        # after tweepy preprocessing the colon symbol left remain after
        # removing mentions
        tweet = re.sub(r':', '', tweet)
        tweet = re.sub(r'‚Ä¶', '', tweet)

        # replace consecutive non-ASCII characters with a space
        tweet = re.sub(r'[^\x00-\x7F]+', ' ', tweet)

        # remove emojis from tweet
        tweet = emojiPattern.sub(r'', tweet)

        # looping through conditions
        filteredTweet = []
        for w in wordTokens:
            # check tokens against stop words , emoticons and punctuations
            if w not in stopWords and w not in emoticons and w not in string.punctuation:
                filteredTweet.append(w)

        return ' '.join(filteredTweet)

    def getTweets(self, location, hashtag, startDate, endDate):

        df = pd.DataFrame(columns=self.cols)

        query = {'fields': [{'items': [hashtag]}],
                 'near': (location, 40),
                 'since': startDate,
                 'until': endDate}

        for status in BlueBird().search(query):
            print("Fetching Tweets :P")
            newEntry = []
            # status = status._json

            # if this tweet is a retweet update retweet count
            if status['created_at'] in df['created_at'].values:
                i = df.loc[df['created_at'] == status['created_at']].index[0]

                cond1 = status['favorite_count'] != df.at[i, 'favorite_count']
                cond2 = status['retweet_count'] != df.at[i, 'retweet_count']
                if cond1 or cond2:
                    df.at[i, 'favorite_count'] = status['favorite_count']
                    df.at[i, 'retweet_count'] = status['retweet_count']

                continue

            # clean the tweet
            cleanText = self.cleanTweets(status['full_text'])

            # Calculate the Sentiment
            blob = TextBlob(cleanText)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity

            newEntry += [status['id'], status['created_at'], status['full_text'], cleanText,
                         polarity, subjectivity, status['favorite_count'], status['retweet_count']]

            try:
                is_sensitive = status['possibly_sensitive']
            except KeyError:
                is_sensitive = None

            newEntry.append(is_sensitive)

            hashtags = ", ".join([hashtag_item['text'] for hashtag_item in status['entities']['hashtags']])
            newEntry.append(hashtags)  # append the hashtags

            #
            mentions = ", ".join([mention['screen_name'] for mention in status['entities']['user_mentions']])
            newEntry.append(mentions)  # append the user mentions

            try:
                location = status['place']['full_name']
            except TypeError:
                location = ''
            newEntry.append(location)

            # now append a row to the dataframe
            single_tweet_df = pd.DataFrame([newEntry], columns=self.cols)
            df = df.append(single_tweet_df, ignore_index=True)

        return df

if __name__ == "__main__":

    hashtags = ["#BreatheLife", "#AirPollution", "#airquality", "#cleanair", "#airpollution", "#pollution", "#hvac",
                "#airpurifier", "#indoorairquality", "#air", "#climatechange", "#indoorair", "#environment",
                "#airconditioning", "#heating", "#freshair", "#airfilter", "#ventilation", "#airconditioner",
                "#airqualityindex", "#pm2_5 ", "#emissions", "#natureishealing", "#nature", "#pollutionfree",
                "#wearethevirus", 'AirPollution', 'Environment', 'Ozone Layer', 'Global Warming', 'Climate Change',
                'Greenhouse Gases', 'Trees', 'Carbon', 'Aerosals', 'Air', 'Save the planet', 'Factories',
                'Inversion', 'Sulfur', 'AIRS', 'ecosystem', 'Hydrochlorofluorocarbon', 'hydrocarbon', 'TAC', 'zero',
                'pollutant', '#air', '#pollution', '#airpollution', '#coal', '#particles', '#smog', '#cleanair',
                '#airqualityindex', '#climatechange', '#airquality', '#globalwarming', '#airpollutionawareness',
                '#airpollutioncontrol', '#CleanEnergy', '#saveearth', 'Hygroscopicity']

    locations = ["Lagos, Nigeria", "Cape Town, South Africa", "Johannesburg, South Africa", "Accra, Ghana",
                 "Nairobi, Kenya", "Mombasa, Kenya", "Kigali, Rwanda", "Kampala, Uganda"]

    acquireTweets = dataAcq()
    frames = []

    datetoday = date.today()
    today = datetoday.strftime("%Y-%m-%d")

    for location in locations:
        df = []
        for hashtag in hashtags:
            tweets = acquireTweets.getTweets(location=location, hashtag=hashtag, startDate="2020-08-01",
                                             endDate=today)
            df.append(tweets)
            conc_df = pd.concat(df, ignore_index=True)
            print(conc_df.shape)
            frames.append(conc_df)

    final_df = pd.concat(frames, ignore_index=True)
    print(final_df.shape)

    final_df.to_csv('postlockdown.csv', index=False, encoding="utf-8")
