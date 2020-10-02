import tweepy
import os
import re
import logging
import nltk
import string
import datetime
import pandas as pd
import numpy as np
import preprocessor as p
from tweepy import OAuthHandler
from tweepy import API
from tweepy import Cursor
from nltk.corpus import stopwords
from textblob import TextBlob


form = logging.Formatter("%(asctime)s : %(levelname)-5.5s : %(message)s")
logger = logging.getLogger()

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(form)
logger.addHandler(consoleHandler)

logger.setLevel(logging.DEBUG)

class dataAcq():
    def __init__(self, cols=None, auth=None):
        if cols is not None:
            self.cols = cols
        else:
            self.cols = ['id', 'created_at', 'source', 'original_text', 'clean_text',
                         'polarity', 'subjectivity', 'lang', 'favorite_count', 'retweet_count', 'original_author',
                         'possibly_sensitive', 'hashtags', 'user_mentions', 'place', 'place_coord_boundaries']

        if auth is None:
            # Variables that contains the user credentials to access Twitter API
            consumer_key = os.environ.get('TWITTER_API_KEY')
            consumer_secret = os.environ.get('TWITTER_API_SECRET')
            access_token = os.environ.get('TWITTER_ACCESS_TOKEN')
            access_token_secret = os.environ.get('TWITTER_ACCESS_TOKEN_SECRET')

            # This handles Twitter authetification and the connection to Twitter Streaming API
            auth = OAuthHandler(consumer_key, consumer_secret)
            auth.set_access_token(access_token, access_token_secret)

        self.auth = auth
        self.api = API(auth, wait_on_rate_limit=True)

    def cleanTweets(self, twitterText):
        logger.info("Cleaning a tweet \n\n")
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

    def getTweets(self, keyword, geocode, startDate, endDate, csvfile):

        df = pd.DataFrame(columns=self.cols)
        if csvfile is not None:
            # If the file exists, then read the existing data from the CSV file.
            if os.path.exists(csvfile):
                df = pd.read_csv(csvfile, header=0)

        for status in Cursor(self.api.search, q=keyword, geocode=geocode, lang="en", since_id=startDate,
                             tweet_mode="extended", include_rts=False).items(200):
            logger.info("Working on new Tweet :)")

            newEntry = []
            status = status._json

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
            cleanText = self.cleanTweets(status['text'])

            # Calculate the Sentiment
            blob = TextBlob(cleanText)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity

            newEntry += [status['id'], status['created_at'], status['source'], status['text'], cleanText,
                         polarity, subjectivity, status['lang'], status['favorite_count'], status['retweet_count']]

            newEntry.append(status['user']['screen_name'])

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
                xyz = status['place']['bounding_box']['coordinates']
                coordinates = [coord for loc in xyz for coord in loc]
            except TypeError:
                coordinates = None
            newEntry.append(coordinates)

            try:
                location = status['user']['location']
            except TypeError:
                location = ''
            newEntry.append(location)

            # now append a row to the dataframe
            single_tweet_df = pd.DataFrame([newEntry], columns=self.cols)
            df = df.append(single_tweet_df, ignore_index=True)

        df.to_csv("micTest12.csv", columns=self.cols, index=False, encoding="utf-8")
        return df


if __name__ == "__main__":

    keywords = ["#airquality", "#cleanair", "#airpollution", "#pollution", "#hvac", "#airpurifier", "#indoorairquality",
                "#climatechange", "#indoorair", "#environment", "#airconditioning", "#coronavirus", "#heating",
                "#ac", "#airfilter", "#allergies", "#hvacservice", "#ventilation", "#wellness", "#delhipollution",
                "#airconditioner", "#airqualityindex", "#bhfyp “particulate matter” “fine particulate matter”", "#air",
                "#pm2_5", "#emissions", "#natureishealing", "#nature", "#pollutionfree", "#wearethevirus", "#freshair",
                "#safety", "#covid", "#health"]
    startDate = datetime.datetime(2019, 4, 1, 0, 0, 0)
    endDate = datetime.datetime(2020, 8, 1, 0, 0, 0)
    geocodes = ["-33.918861,18.423300,40km", "-26.205681,28.046822,40km", "5.550000,-0.020000,40km",
                "-1.286389,36.817223,40km", "-4.043740,39.658871,40km", "6.465422,3.406448,40km",
                " -1.935114,30.082111,40km", "0.347596,32.582520,40km"]
    csvfile = "micTest12.csv"

    acquireTweets = dataAcq()
    for keyword in keywords:
        for geocode in geocodes:
            df = acquireTweets.getTweets(keyword=keyword, geocode=geocodes, startDate=startDate, endDate=endDate,
                                         csvfile=csvfile)

    print(df.head())
