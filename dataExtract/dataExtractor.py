from abc import ABC, abstractmethod
import csv
import os
import numpy as np
import re
import pandas as pd
import pickle
from datetime import datetime
from datetime import timedelta
import math


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

class FinExtractor:
    _pickleL = '../resources/processed/Pickle_NASDAQ.p'
    def FinFormat(self, filename):
        """
        Used to remove spaces from excel file and wrote to CSV
        :param filename:
        :return:
        """
        df = pd.read_excel(filename)
        df1 = df.dropna(axis=1, how='all')
        newFP = os.path.split(filename)[0] + '/CSV_NASDAQ.csv'
        df1.to_csv(path_or_buf=newFP)


    def FinExtractAndPickle(self, filename, pickleLoc):
        """
        Turn into MultiIndex form after manual (scripted) tampering in CSV format, then store in pickle form
        :param filename:
        :return:
        """
        df = pd.read_csv(filename, float_precision='high', header=[0,1,2])

        #print(df.loc[(slice(None), slice(None)), 'Date'])
        pickle.dump(df, open(pickleLoc, 'wb'))

    def FinConvExtractfromPickle(self, pickleLoc):
        """
        Was useful for not having to load in the csv file every time to bugfix the dataframe
        :param pickleLoc:
        :return: dataframe of financial data
        """
        df = pickle.load(open(pickleLoc, 'rb'))
        levels = df.dtypes.index.levels
        for l1 in levels[0]:
            for l2 in levels[1]:
                df[l1, l2, 'Date'] = df[l1,l2,'Date'].apply(ConvertExcelTime, convert_dtype=True)
        convertedLoc = os.path.splitext(pickleLoc)[0] + '_convertedDate.p'
        pickle.dump(df, open(convertedLoc, 'wb'))
        return convertedLoc

    def ConvertExcelTime(self, days):
        if(math.isnan(days)):
            return math.nan
        dt = datetime.fromordinal(datetime(1900, 1, 1).toordinal() + math.floor(days) - 2)
        dec = days % 1
        return dt + timedelta(days=dec)

    def FinExtractFromPickle(self, pickleLoc):
        return pickle.load(open(pickleLoc, 'rb'))