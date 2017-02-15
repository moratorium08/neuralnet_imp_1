# coding:utf-8

import yaml
import re
from tweepy import OAuthHandler, API

keys = yaml.load(open("twitter_keys", "r"))
consumer_key = keys["consumer_key"]
consumer_secret = keys["consumer_secret"]
access_token = keys["access_token"]
access_secret = keys["access_secret"]

handler = OAuthHandler(consumer_key, consumer_secret)
handler.set_access_token(access_token, access_secret)

api = API(handler)

filename = "users"

with open(filename, "r") as f:
    usernames = f.read().strip("\n").split("\n")

print("use the following users...")
for name in usernames:
    print(name)


with open("tweets", "w") as f:
    for username in usernames:
        tweets = api.user_timeline(screen_name=username, count=200)
        tweets = map(lambda x: re.sub("@[A-Za-z0-9_]+", "", x.text), tweets)
        tweets = map(lambda x: x.strip(" ").replace("\n", ""), tweets)
        f.write('\n'.join(tweets))
