import pandas as pd
import numpy as np
import tweepy, json

import EngineFiles.apiAuth as au
import EngineFiles.TweetStreamer as ts

def GenerateTweets(max_tweets_data = 100,keyword=['tweet'], languages=[]):
    api  = au.apiAuth()
    auth = tweepy.OAuthHandler(api.consumer_key, api.consumer_secret)
    auth.set_access_token(api.access_token, api.access_token_secret)

    listener = ts.MyStreamListener(max_tweets=max_tweets_data)
    stream = tweepy.Stream(auth, listener, tweet_mode='extended')
    if len(languages) > 0:
        try:
            stream.filter(track=keyword,languages=languages)
        except:
            print('Something wrong while retrieving tweets from API!')
    else:
        try:
            stream.filter(track=keyword)
        except:
            print('Something wrong while retrieving tweets from API!')
    print('All tweets already generated!')

def GetTweets():
    try:
        tweets_file = open('Data/tweets.txt', 'r')
    except:
        return 'Something wrong with the collected data!'

    tweets_data = []

    for line in tweets_file:
        tweet = json.loads(line)
        tweets_data.append(tweet)

    tweets_file.close()

    tweet_df = pd.DataFrame(tweets_data, columns=['text', 'extended_tweet', 'retweeted_status','lang'])
    tweet_df = tweet_df[tweet_df['lang']=='in']
    tweet_df.drop(labels=['lang'], axis=1, inplace=True)

    tweet_df.extended_tweet = tweet_df.extended_tweet.replace(np.nan, 0)
    tweet_df.retweeted_status = tweet_df.retweeted_status.replace(np.nan, 0)

    tweet_df['full_tweet1'] = tweet_df.extended_tweet.apply(lambda x : x['full_text'] if x != 0 else 0)
    tweet_df['full_tweet2'] = tweet_df.retweeted_status.apply(lambda x : x['extended_tweet']['full_text'] if type(x) != int and 'extended_tweet' in x.keys() else 0)
    for i,v in tweet_df.transpose().items():
        if v['full_tweet1'] != 0:
            tweet_df['text'][i] = v['full_tweet1']

        if v['full_tweet2'] != 0:
            tweet_df['text'][i] = v['full_tweet2']

    tweet_df.drop(['extended_tweet'], axis=1, inplace=True)
    tweet_df.drop(['retweeted_status'], axis=1, inplace=True)
    tweet_df.drop(['full_tweet1'], axis=1, inplace=True)
    tweet_df.drop(['full_tweet2'], axis=1, inplace=True)

    return tweet_df