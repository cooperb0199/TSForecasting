# -*- coding: utf-8 -*-
import tweepy
from tweepy import OAuthHandler
from Repositories.rawTwitterObject import RawTwitterObject

# Intents to find:
#   exchange, partnership, wallet, application launch, roadmap, announcement
#   ahead of schedule, bullish trends, rebrand
#



class TwitterAPIClient:
    consumer_key = ''
    consumer_secret = ''
    access_token = ''
    access_secret = ''
     
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
     
    api = tweepy.API(auth)
    id2text = []
    
    def GetTweets(self, identifier, nickname, amountOfItems):
        for status in tweepy.Cursor(self.api.user_timeline, id=identifier).items(amountOfItems):
#            print(status._json)
            self.id2text.append(RawTwitterObject(identifier, nickname, status.id, status.text, status.created_at))

