from got3 import models, manager
import time
import re
import string
from datetime import datetime
from nltk.corpus import wordnet
import csv
import random

usernames = [
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
]

numColumns_labelled = 4

translateDict = {
    '-1':'neg',
    '0':'neutral',
    '1':'pos'
}

ascii = set(string.printable)


def filterASCII(sentence):
    return ''.join(filter(lambda x: x in ascii, sentence))
def getFromServer():
    pattern = re.compile(r"http\S+|pic\.twitter\.com\S+|tus\/\d\S+|http[s]?\/\/[www\.]?twitter\.com\S+\s\d+")
    delchars = str.maketrans(dict.fromkeys(''.join(c for c in map(chr, range(256)) if not (c.isalnum() or c.isspace()))))

    with open('../resources/hydrated_tweets/EnglishFilterBatch.txt', 'a', encoding='utf-8') as dump:
        for user in usernames:
            timeBefore = time.time()
            print('[%s]' %(datetime.now().isoformat(sep=' ')), ' Getting: ', user)
            tweetCriteria = manager.TweetCriteria().setUsername(user).setSince("2015-10-03")
            tweets = manager.TweetManager.getTweets(tweetCriteria)
            print('Time took for: %s ' %(user), time.time() - timeBefore)
            print('Num of Tweets: ', len(tweets))
            for tweet in tweets:
                dump.write(tweet.date.isoformat(sep=' '))
                dump.write('\t')
                dump.write(user)
                dump.write('\t')

                textToWrite = filterASCII(re.sub('\d{6,}$', '', pattern.sub("", tweet.text.replace(':// ', '://')).translate(delchars).lower().strip()))
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
                if(random.random() <= ratio):
                    subFile.write(line)

def filterSubSample(fileName, outFileName):
    with open(fileName, 'r') as unFil:
        with open(outFileName, 'w') as fil:
            for line in unFil:
                if(len(line.split('\t')) > 3):
                    fil.write(line)



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
    combineData('../resources/hydrated_tweets/FilteredSubsampled_TwitterData_v3.txt', '../resources/hydrated_tweets/TwitterData_v3.txt', '../resources/hydrated_tweets/Combined_TwitterData_v3.txt')