import tweepy, json

class MyStreamListener(tweepy.StreamListener) :
    def __init__(self, max_tweets, api=None) :
        super(MyStreamListener, self).__init__()
        self.num_tweets = 0
        self.file = open("Data/tweets.txt","w")
        self.max_tweets = max_tweets

    def on_status(self, status) :
        if hasattr(status, "extended_tweet"):
            full_tweet = status.extended_tweet["full_text"]
        else:
            full_tweet = status.text
        tweet = status._json
        self.file.write(json.dumps(tweet) + '\n')
        self.num_tweets += 1
        if self.num_tweets < self.max_tweets :
            return True
        else :
            return False
        self.file.close()
        self.file_full.close()

    def on_error(self, status) :
        print(status)