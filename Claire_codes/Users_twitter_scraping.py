#!/usr/bin/env python
# coding: utf-8

# In[1]:


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



# In[2]:


from requests import get
from requests.exceptions import RequestException
from contextlib import closing
# from bs4 import BeautifulSoup
import pandas as pd
import os, sys

#import fire

import sys
import os
import json
import matplotlib.pyplot as plt
import re
import string

import matplotlib.dates as mdates
import seaborn as sns

# to view all columns
pd.set_option("display.max.columns", None)

import tweepy
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy import API
from tweepy import Cursor
from datetime import datetime, date, time, timedelta
from collections import Counter
import sys

import preprocessor as p


# In[3]:


#Variables that contains the user credentials to access Twitter API 
consumer_key = ''
consumer_secret = ''
access_token = ''
access_token_secret =''
sleep_on_rate_limit=False

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
auth_api = API(auth, wait_on_rate_limit=True)


# In[ ]:


hashtags=["#BreatheLife","#AirPollution","#airquality","#cleanair","#airpollution","#pollution","#hvac","#airpurifier" ,"#indoorairquality","#air","#climatechange","#indoorair","#environment" ,"#airconditioning","#heating" , "#freshair", "#airfilter","#ventilation","#airconditioner","#airqualityindex", "#pm2_5 ","#emissions","#natureishealing","#nature","#pollutionfree" ,"#wearethevirus",'AirPollution', 'Environment', 'Ozone Layer', 'Global Warming', 'Climate Change', 'Greenhouse Gases', 'Trees', 'Carbon',
        'Aerosals', 'Air', 'Save the planet', 'Factories', 'Hygroscopicity', 'Inversion', 'Sulfur', 'AIRS', 'ecosystem', 'Hydrochlorofluorocarbon',
        'hydrocarbon', 'TAC', 'zero', 'pollutant', '#air', '#pollution', '#airpollution', '#coal', '#particles', '#smog', '#cleanair',
       '#airqualityindex', '#climatechange', '#airquality', '#globalwarming', '#airpollutionawareness', '#airpollutioncontrol',
       '#CleanEnergy', '#saveearth']

geocodes=["6.48937,3.37709,500km","-33.99268,18.46654,500km","-26.22081,28.03239,500km","5.58445,-0.20514,500km","-1.27467,36.81178,500km","-4.04549,39.66644,500km","-1.95360,30.09186,500km","0.32400,32.58662,500km"]

names = []


names = []
for hashtag in hashtags:
    for geocode in geocodes:
        tweets = tweepy.Cursor(auth_api.search, q = hashtag ,geocode=geocode).items(12500)
        users = []
        for status in tweets:
            name = status.user.screen_name
            t=status.text
            users.append(name)
        names.append(users)
screen_names = [y for x in names for y in x]
screen_names


# In[12]:


screen_names
df = pd.DataFrame(screen_names)
df.to_csv("handles_new.csv")


# In[ ]:




