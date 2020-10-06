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

# Variables that contains the user credentials to access Twitter API
consumer_key = 'JKNh3lOfHZOtkuhDmlJPAB6sy'
consumer_secret = 'W22BOhcbcDiqe8deTnnZCFIHJu7Zk3HLKoNpAQZXoFink94ujp'
access_token = '2464951318-BbgDuJQPLWXB9Ad3qyQXIElTZuVseIujax47mLK'
access_token_secret = '3siNawNyHuyUuM9tCOfWe57yobYWu2arYSkurJ1xwxVk1'
sleep_on_rate_limit = False

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
auth_api = API(auth, wait_on_rate_limit=True)

hashtags = ["#BreatheLife", "#AirPollution", "#airquality", "#cleanair", "#airpollution", "#pollution", "#hvac",
            "#airpurifier", "#indoorairquality", "#air", "#climatechange", "#indoorair", "#environment",
            "#airconditioning", "#heating", "#freshair", "#airfilter", "#ventilation", "#airconditioner",
            "#airqualityindex", "#pm2_5 ", "#emissions", "#natureishealing", "#nature", "#pollutionfree",
            "#wearethevirus", 'AirPollution', 'Environment', 'Ozone Layer', 'Global Warming', 'Climate Change',
            'Greenhouse Gases', 'Trees', 'Carbon', 'Aerosals', 'Air', 'Save the planet', 'Factories', 'Hygroscopicity',
            'Inversion', 'Sulfur', 'AIRS', 'ecosystem', 'Hydrochlorofluorocarbon', 'hydrocarbon', 'TAC', 'zero',
            'pollutant', '#air', '#pollution', '#airpollution', '#coal', '#particles', '#smog', '#cleanair',
            '#airqualityindex', '#climatechange', '#airquality', '#globalwarming', '#airpollutionawareness',
            '#airpollutioncontrol', '#CleanEnergy', '#saveearth']

geocodes = ["6.48937,3.37709,500km", "-33.99268,18.46654,500km", "-26.22081,28.03239,500km", "5.58445,-0.20514,500km",
            "-1.27467,36.81178,500km", "-4.04549,39.66644,500km", "-1.95360,30.09186,500km", "0.32400,32.58662,500km"]
subgeocodes = ["-1.27467,36.81178,500km", "-4.04549,39.66644,500km"]
names = []


names = []
for hashtag in hashtags:
    for geocode in subgeocodes:
        tweets = tweepy.Cursor(auth_api.search, q=hashtag, geocode=geocode).items(12500)
        users = []
        for status in tweets:
            name = status.user.screen_name
            t = status.text
            users.append(name)
        names.append(users)

        screen_names = [y for x in names for y in x]

        df = pd.DataFrame(screen_names)
        df.to_csv(geocode + "_12500.csv")
