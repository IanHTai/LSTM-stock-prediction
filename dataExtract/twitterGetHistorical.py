from got3 import models, manager
import time
import re
import string
from datetime import datetime
from nltk.corpus import wordnet
import csv
import random

usernames = [
    'Reuters'
]

"""
'AJEnglish',
    'ABCPolitics',
    'AP',
    'AP_Politics',
    'BBCNews',
    'BBCPolitics',
    'BBCScienceNews',
    'CBSNews',
    'CNN',
    'OnTheMoneyCBC',
    'FoxNews',
    'MSNBC',
    'Reuters',
    'WSJFinReg',
    'WSJ',
    'FT',
    'FinancialTimes',
    'nytimes',
    'BBCBreaking',
    'BreakingNews',
    'YahooFinance',
    'YahooNews',
    'CSPAN',
    'WSJbreakingnews',
    'TheEconomist',
    'CNNMoney',
    'politico',
    'washingtonpost',
    'NBCFirstRead',
    'CNNPolitics',
    'ReutersLive',
    'SkyNewsBreak',
    'CBSTopNews',
    'ajenews',
    'cnni',
    'CNNSitRoom',
    'arstechnica',
"""

numColumns_labelled = 4

translateDict = {
    '-1':'neg',
    '0':'neutral',
    '1':'pos'
}

keyWords = [
    'aapl',
    'apple',
    'microsoft',
    'msft',
    'google',
    'googl',
    'goog',
    'alphabet',
    'amazon',
    'amzn'
]

ascii = set(string.printable)


def filterASCII(sentence):
    return ''.join(filter(lambda x: x in ascii, sentence))
def getFromServer(fileName):
    pattern = re.compile(r"http\S+|pic\.twitter\.com\S+|tus\/\d\S+|http[s]?\/\/[www\.]?twitter\.com\S+\s\d+")
    delchars = str.maketrans(dict.fromkeys(''.join(c for c in map(chr, range(256)) if not (c.isalnum() or c.isspace()))))

    with open(fileName, 'a', encoding='utf-8') as dump:
        for user in usernames:
            timeBefore = time.time()
            print('[%s]' %(datetime.now().isoformat(sep=' ')), ' Getting: ', user)
            tweetCriteria = manager.TweetCriteria().setUsername(user).setSince("2017-06-20").setUntil("2017-12-01")
            tweets = manager.TweetManager.getTweets(tweetCriteria)
            print('Time took for: %s ' %(user), time.time() - timeBefore)
            print('Num of Tweets: ', len(tweets))
            for tweet in tweets:
                textToWrite = filterASCII(re.sub('\d{6,}$', '',
                                                 pattern.sub("", tweet.text.replace(':// ', '://')).translate(
                                                     delchars).lower().strip()))
                if not (any(substring in textToWrite for substring in keyWords)):
                    continue

                dump.write(tweet.date.isoformat(sep=' '))
                dump.write('\t')
                dump.write(user)
                dump.write('\t')


                dump.write(textToWrite)
                dump.write('\n')

#Convert old non-punctuation filtered file
def removeEnglish(inFileName, outFileName):
    delchars = str.maketrans(dict.fromkeys(''.join(c for c in map(chr, range(256)) if not (c.isalnum() or c.isspace()))))
    counter = 1
    with open(inFileName, 'r') as inFile:
        with open(outFileName, 'a') as outFile:
            for line in inFile:
                print(counter)
                counter += 1
                splitLine = line.split('\t')
                for i in range(len(splitLine) - 1):
                    outFile.write(splitLine[i])
                    outFile.write('\t')
                newText = re.sub('\d{6,}$','',splitLine[-1].translate(delchars).lower().strip())
                #for word in splitLine[-1].split():
                #    if wordnet.synsets(word):
                #        outFile.write(word)
                #        outFile.write(' ')
                outFile.write(newText)
                outFile.write('\n')
def removeLeftoverURLParts(fileName):
    blacklist = []
    with open('../resources/hydrated_tweets/TwitterData_v3.txt', 'w') as outFile:
        with open(fileName, 'r') as csvFile:
            df = csv.reader(csvFile, delimiter='\t')
            for row in df:
                if row[1] == 'arstechnica':
                    for word in row[2].split():
                        if len(word) > 25:
                            blacklist.append(word)
        with open(fileName, 'r') as csvFileAgain:
            #pattern = '|'.join('(%s)' % re.escape(word) for word in blacklist)
            for line in csvFileAgain:
                splitLine = line.split('\t')
                tempSplitLine2 = splitLine[2]
                for word in splitLine[2].split():
                    if 'youtubecomwatch' in word:
                        tempSplitLine2 = re.sub('youtubecom\S+\s\S+', '', tempSplitLine2)
                    elif len(word) > 25 or (len(word) > 15 and (sum(c.isdigit() for c in word) + 1)/(sum(c.isalpha() for c in word) + 1) >= 0.3):
                        tempSplitLine2 = tempSplitLine2.replace(word, '')
                outFile.write(splitLine[0])
                outFile.write('\t')
                outFile.write(splitLine[1])
                outFile.write('\t')
                outFile.write(tempSplitLine2)

def subSample(desiredNum, inFileName, outFileName):
    count = 0
    with open(inFileName, 'r') as readFile:
        for line in readFile:
            count+=1
    ratio = desiredNum/float(count)
    with open(outFileName, 'w') as subFile:
        with open(inFileName, 'r') as readFile:
            for line in readFile:
                if len(line.split('\t')) > 2:
                    if random.random() <= ratio:
                        subFile.write(line)




def combineData(labelledPath, unlabelledPath, outPath):
    with open(labelledPath, 'r') as labelled:
        with open(unlabelledPath, 'r') as unlabelled:
            with open(outPath, 'w') as out:
                for line in labelled:
                    splitLine = line.rstrip().split('\t')
                    if(len(splitLine) == numColumns_labelled):
                        splitLine[3] = translateDict[splitLine[3]]
                        writeLine = '\t'.join(splitLine)
                        out.write(writeLine)
                        out.write('\n')
                    else:
                        out.write(line.rstrip())
                        out.write('\t-1\n')
                for line in unlabelled:
                    out.write(line.rstrip())
                    out.write('\t-1\n')
    return outPath
if __name__ == '__main__':
    '''subFileName = '../resources/hydrated_tweets/small_data/Subsampled_Dev.txt'
    subSample(5000, '../resources/hydrated_tweets/TwitterData_v3.txt', subFileName)

    combineData('../resources/hydrated_tweets/FilteredSubsampled_TwitterData_v3.txt', subFileName, '../resources/hydrated_tweets/small_data/Combined_subsampled_Dev.txt')'''

    #subSample(100, '../resources/hydrated_tweets/TwitterData_v3.txt', '../resources/hydrated_tweets/small_data/Test_Dev.txt')

    getFromServer('../resources/hydrated_tweets/relevant_tweets.txt')