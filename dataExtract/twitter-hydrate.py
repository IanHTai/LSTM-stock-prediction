import requests
import twitter
import re

api = twitter.Api(consumer_key='PXWcQK3KxbqPJgTVbNL4n8E8u',
                      consumer_secret='AC7wMzSsERkiPrIjd7c0Uds34UnPSG68gW1rVVf1SX7RqlCUVk',
                      access_token_key='1969776098-gRrecm9nFjc2OcY2yMKgNdWjPttFMJy3FpNhBgh',
                      access_token_secret='2QcIDg2RCu2c9gXfKw0jdMpmxpmDcxhr8v8DkkiVhz1eW')
with open('../resources/hydrated_tweets/batch1.txt', 'w', encoding='utf-8') as dump:
    with open('../resources/raw/news_outlets.txt', 'r') as twids:
        for i in range(0, 500):
            try:
                dump.write(re.sub(r"http\S+", "", api.GetStatus(twids.readline()).text))
                dump.write('\n')
            except twitter.TwitterError:
                print('twitter error')