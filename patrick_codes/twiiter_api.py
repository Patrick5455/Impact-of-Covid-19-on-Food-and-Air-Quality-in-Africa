#!/usr/bin/env python
# coding: utf-8

# ## This Notebook would be used to Fetch Twitter Data via Twitter API

# > import libraries

# In[55]:


import pandas as pd
import tweepy
from tweepy import OAuthHandler
from tweepy import API  
from tweepy import Cursor
from datetime import datetime, date, time, timedelta
from collections import Counter
import os, sys
import csv


# > Load dotenv to expose api keys to the application

# In[57]:


from dotenv import load_dotenv
load_dotenv('../.env')


# In[58]:


API_KEY="API_KEY"
API_SECRET_KEY="API_SECRET_KEY"
ACCESS_TOKEN="ACCESS_TOKEN"
ACCESS_TOKEN_SECRET="ACCESS_TOKEN_SECRET"
print(API_KEY, API_SECRET_KEY, ACCESS_TOKEN, ACCESS_TOKEN_SECRET)


# In[59]:


API_KEY = os.environ.get(API_KEY)
API_SECRET_KEY = os.getenv(API_SECRET_KEY)
ACCESS_TOKEN = os.getenv(ACCESS_TOKEN)
ACCESS_TOKEN_SECRET=os.getenv(ACCESS_TOKEN_SECRET)


# In[60]:


auth = OAuthHandler(API_KEY, API_SECRET_KEY)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth, wait_on_rate_limit=True)
auth_api = API(auth)


# > Testing Api

# In[63]:


search_words = "airquality"
date_since="2020-03-03"  
# Collect tweets
tweets = tweepy.Cursor(api.search,
              q=search_words,
              lang="en", 
              since=date_since).items(2)
# Iterate and print tweets
for tweet in tweets:
    print(tweet.text)


# > TWITTER API ALL SET UP!

# In[ ]:




