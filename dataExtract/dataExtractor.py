from abc import ABC, abstractmethod
import csv
import os
import numpy as np
import re
import pandas as pd


class DataExtractor(ABC):
    @abstractmethod
    def __init__(self, filename):
        pass

    @abstractmethod
    def process(self):
        pass

class MichiganExtractor(DataExtractor): #Used for the Michigan tweet dataset, for the format [score\ttweet]
    def __init__(self, filename):
        self.filename = filename

    def process(self):
        tempList = list(csv.reader(open(self.filename, 'r', encoding='utf-8'), delimiter='\t'))
        self.fileDict = {}
        for tweet in tempList:
            if int(tweet[0]) in self.fileDict:
                self.fileDict[int(tweet[0])].append(tweet[1])
            else:
                self.fileDict[int(tweet[0])] = [tweet[1]]
        return self.fileDict

class ConVoteExtractor(DataExtractor): #Used for congressional vote dataset
    def __init__(self, filename):
        self.filepath = filename

    def process(self):
        self.fileArr = []
        for filename in os.listdir(self.filepath):
            with open(self.filepath + '\\' + filename, 'r', encoding='utf-8') as singleSpeech:
                if filename.split('.')[0].strip()[-1] == 'N':
                    self.fileArr.append([singleSpeech.read(), 0])
                elif filename.split('.')[0].strip()[-1] == 'Y':
                    self.fileArr.append([singleSpeech.read(), 1])
                else:
                    print('No Vote Found: ', singleSpeech.read())
        return np.array(self.fileArr)
    def cleanData(self, outPath):
        delchars = str.maketrans(
            dict.fromkeys(''.join(c for c in map(chr, range(256)) if not (c.isalnum() or c.isspace()))))

        for filename in os.listdir(self.filepath):
            text = ''
            with open(self.filepath + '\\' + filename, 'r', encoding='utf-8') as singleSpeech:
                text = singleSpeech.read()
                with open(outPath + '\\' + filename, 'w', encoding='utf-8') as singleSpeech:
                    noSpaceApo = re.sub('\s(?=[^i]{0,1}\'\S+)', '', text)
                    toWrite = noSpaceApo.translate(delchars).lower().strip()
                    singleSpeech.write(toWrite)

class RTPolarityExtractor(DataExtractor): #Used for the RT-PolarityData set
    def __init__(self, filename):
        self.filepath = filename

    def process(self):
        delchars = str.maketrans(
            dict.fromkeys(''.join(c for c in map(chr, range(256)) if not (c.isalnum() or c.isspace()))))
        self.fileArr = []
        for filename in os.listdir(self.filepath):
            if os.path.splitext(filename)[1] == 'pos':
                with open(self.filepath + '\\' + filename) as posFile:
                    for line in posFile:
                        toWrite = line.translate(delchars).lower().strip()
                        self.fileArr.append([toWrite, 1])

            elif os.path.splitext(filename)[1] == 'neg':
                with open(self.filepath + '\\' + filename) as negFile:
                    for line in negFile:
                        toWrite = line.translate(delchars).lower().strip()
                        self.fileArr.append([toWrite, 0])
            else:
                raise FileNotFoundError()
        return np.array(self.fileArr)

def GoogleFinExtract(stockNames):
    for i in stockNames:
        data = pd.read_csv('http://www.google.com/finance/getprices?i=300&p=100d&f=d,o,h,l,c,v&df=cpct&q=' + i, skiprows=8, header=None)
        print(data.head())

if __name__ == '__main__':
    GoogleFinExtract(['AAPL'])