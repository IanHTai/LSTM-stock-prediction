from abc import ABC, abstractmethod
import csv
import os


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
        self.fileDict = {0: [], 1: []}
        for filename in os.listdir(self.filepath):
            with open(self.filepath + '\\' + filename, 'r', encoding='utf-8') as singleSpeech:
                if filename.split('.')[0].strip()[-1] == 'N':
                    self.fileDict[0].append(singleSpeech.read())
                elif filename.split('.')[0].strip()[-1] == 'Y':
                    self.fileDict[1].append(singleSpeech.read())
                else:
                    print('No Vote Found: ', singleSpeech.read())
        return self.fileDict