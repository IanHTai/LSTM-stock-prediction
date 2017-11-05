import requests
import twitter
import re
import time
import json
from twitter.twitter_utils import enf_type
import atexit
import string
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from datetime import datetime
import os

start = 123300 #Input this every time

ascii = set(string.printable)

twitterDateTimeFormat = '%a %b %d %H:%M:%S %z %Y'
limitRemain = 900
limitReset = 0

class multiStatusTwitter(twitter.Api):
    def getMultiStatus(self, status_ids,
                  trim_user=False,
                  include_my_retweet=True,
                  include_entities=True,
                  include_ext_alt_text=True):

        """
        Function for retrieving up to 100 statuses
        Will also filter non-english tweets

        :param status_ids: list of status IDs
        :param trim_user:
        :param include_my_retweet:
        :param include_entities:
        :param include_ext_alt_text:
        :return: Parsed list of statuses
        """
        url = '%s/statuses/lookup.json' % (self.base_url)
        stringIds = ','.join(status_ids)
        parameters = {
            'id': stringIds,
            'trim_user': enf_type('trim_user', bool, trim_user),
            'include_my_retweet': enf_type('include_my_retweet', bool, include_my_retweet),
            'include_entities': enf_type('include_entities', bool, include_entities),
            'include_ext_alt_text': enf_type('include_ext_alt_text', bool, include_ext_alt_text)
        }

        resp = self._RequestUrl(url, 'GET', data=parameters)
        if resp.status_code == 200:
            data = self._ParseAndCheckTwitter(resp.content.decode('utf-8'))
            limitRemain = resp.headers._store['x-rate-limit-remaining']
            limitReset = resp.headers._store['x-rate-limit-reset']
            filteredData = []
            for status in data:
                filteredText = re.sub(r"http\S+", "", status['text']).replace('\n', ' ').strip()
                try:
                    if(detect(status['text']) == 'en'):
                        status['text'] = self.filterASCII(filteredText)
                        dateFormatted = datetime.strptime(status['created_at'], twitterDateTimeFormat)
                        status['created_at'] = dateFormatted.isoformat(sep=' ')
                        filteredData.append(status)
                except LangDetectException:
                    pass
            if(limitRemain == 0):
                print('Waiting ', limitReset - time.time() + 15 , ' seconds')
                time.sleep(limitReset - time.time() + 15)
            return filteredData
        else:
            print('Response status code: ', resp.status_code)
            time.sleep(915)
            return self.getMultiStatus(status_ids, trim_user, include_my_retweet, include_entities, include_ext_alt_text)

    def filterASCII(self, sentence):
        return ''.join(filter(lambda x: x in ascii, sentence))

api = multiStatusTwitter(consumer_key='PXWcQK3KxbqPJgTVbNL4n8E8u',
                      consumer_secret='AC7wMzSsERkiPrIjd7c0Uds34UnPSG68gW1rVVf1SX7RqlCUVk',
                      access_token_key='1969776098-gRrecm9nFjc2OcY2yMKgNdWjPttFMJy3FpNhBgh',
                      access_token_secret='2QcIDg2RCu2c9gXfKw0jdMpmxpmDcxhr8v8DkkiVhz1eW')

def exit_handler(index):
    """
    For when the Twitter Hydrater needs to be paused/stopped
    :return:
    """
    print('Reader stopped at: ', index)

totalReadCount = start
currentReadCount = 0

atexit.register(exit_handler, totalReadCount)

with open('../resources/hydrated_tweets/batch1.txt', 'a', encoding='utf-8') as dump:
    with open('../resources/raw/news_outlets.txt', 'r') as twids:
        print('checkPoint 1')
        for i in range(0, start):
            twids.readline()

        while(True):
            ids = []
            for i in range(0, 100):
                line = twids.readline()
                if not line == '':
                    ids.append(line.rstrip('\n'))
                else:
                    print('line number ', i + totalReadCount, ', end of file')
                    sys.exit(0)
            statuses = api.getMultiStatus(ids)
            for status in statuses:
                dump.write(status['created_at'])
                dump.write('\t')
                dump.write(status['user']['id_str'])
                dump.write('\t')
                dump.write(status['user']['name'])
                dump.write('\t')
                dump.write(status['id_str'])
                dump.write('\t')
                dump.write(status['text'])
                dump.write('\n')
            totalReadCount += 100

            print(totalReadCount)