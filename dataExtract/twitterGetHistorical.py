from got3 import models, manager
import time
usernames = [
    'AJEnglish',
    'ABCPolitics',
    'AP',
    'AP_Politics',
    'BBCNews',
    'BBCPolitics',
    'BBCScienceNews',
    'BBCScienceNews',
    'CBSNews',
    'CNN',
    'OnTheMoneyCBC',
    'FoxNews',
    'MSNBC',
    'MSNBC_Politics',
    'NBCPolitics',
    'RT_com',
    'Reuters',
    'TheEconomist',
    'WSJ',
    'arstechnica',
    'guardian',
    'latimes',
    'nhk_news',
    'nytimes',
    'nytscience_',
    'politico,'
    'washingtonpost',
    'NBCFirstRead'
    'CNNPolitics',
    'FRANCE24',
    'Sputniklnt',
    'SputnikNewsUS',
    'telegraphnews',
    'BBCBreaking',
    'BreakingNews',
    'YahooFinance',
    'YahooNews',
    'WSJFinReg',
    'ajenews',
    'cnni',
    'CNNSitRoom',
    'CNNMoney',
    'FT',
    'FinancialTimes',
    'CSPAN',
    'WSJbreakingnews',
    'ReutersLive',
    'SkyNewsBreak',
    'CBSTopNews'
]
timeBefore = time.time()
tweetCriteria = manager.TweetCriteria().setUsername(usernames[0]).setSince("2016-11-04")
tweets = manager.TweetManager.getTweets(tweetCriteria)
print(time.time() - timeBefore)
print(len(tweets))
print(tweet.date.isoformat(sep=' ') for tweet in tweets)