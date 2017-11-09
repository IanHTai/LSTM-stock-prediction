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
import sys


start = 2238300 #Input this every time


filterList = [
    4970411, #Al Jazeera English
    16815644, #ABCPolitics
    51241574, #AP
    426802833, #AP_Politics
    612473, #BBCNews
    621533, #BBC Politics
    621573, #BBC Science News
    621583, #BBC Tech
    15012486, #CBS News
    759251, #CNN
    115505144, #OnTheMoneyCBC
    1367531, #FoxNews
    2836421, #MSNBC
    380075338, #MSNBC Pol
    11856032, #NBC Pol
    64643056, #RT
    1652541, #Reuters
    5988062, #Economist
    3108351, #WSJ
    717313, #arstechnica
    87818409, #guardian
    16664681, #latimes
    204245399, #nhk
    807095, #NYT
    268392524, #NYTSci
    9300262, #politico
    2467791, #WaPo
    52419540, #NBCFirstRead
    13850422, #CNNPol
    1994321, #france24
    3386207949, #SputnikInt
    115618332, #sputnik us
    14138785, #Telegraph news
    5402612, #BBC Breaking
    6017542, #Breaking News
    19546277, #Yahoo Finance
    702881258641051648, #WSJ Financial Regulation
    4898091, #Financial Times
    16311797, #WSJ Politics
    16334857, #WSJ Econ
    18424289, #Al Jazeera News
    2097571, #CNN International
    34310801, #CNN Situation Room
    16184358, #CNN Money
    18949452, #FT
    15675138, #CSPAN
]
filterSet = set(filterList)



ascii = set(string.printable)

twitterDateTimeFormat = '%a %b %d %H:%M:%S %z %Y'

class multiStatusTwitter(twitter.Api):
    limitRemain = 900
    limitReset = 0
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
        try:
            resp = self._RequestUrl(url, 'GET', data=parameters)
        except TimeoutError:
            time.sleep(60)
            return self.getMultiStatus(status_ids, trim_user, include_my_retweet, include_entities,
                                       include_ext_alt_text)
        if resp.status_code == 200:
            data = self._ParseAndCheckTwitter(resp.content.decode('utf-8'))
            if 'x-rate-limit-remaining' in resp.headers._store:
                self.limitRemain = resp.headers._store['x-rate-limit-remaining']
            if 'x-rate-limit-reset' in resp.headers._store:
                self.limitReset = resp.headers._store['x-rate-limit-reset']
            filteredData = []
            for status in data:
                if not (status['user']['id'] in filterSet):
                    continue
                filteredText = re.sub(r"http\S+", "", status['text']).replace('\n', ' ').strip()
                try:
                    if(detect(status['text']) == 'en'):
                        status['text'] = self.filterASCII(filteredText)
                        dateFormatted = datetime.strptime(status['created_at'], twitterDateTimeFormat)
                        status['created_at'] = dateFormatted.isoformat(sep=' ')
                        filteredData.append(status)
                except LangDetectException:
                    pass
            if(self.limitRemain == 0):
                print('Waiting ', self.limitReset - time.time() + 15 , ' seconds')
                time.sleep(self.limitReset - time.time() + 15)
                self.limitRemain == 1
            return filteredData
        else:
            print('Response status code: ', resp.status_code)
            print(resp.reason)
            time.sleep(60)
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