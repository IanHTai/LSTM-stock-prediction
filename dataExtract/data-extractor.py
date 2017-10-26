from abc import ABC, abstractmethod
import csv
import string
from nltk.corpus import stopwords
import re
from nltk.tokenize import word_tokenize
ENGLISH_STOPWORDS = set(stopwords.words('english'))
NEG_CONTRACTIONS = [
    ('arent', 'are not'),
    ('cant', 'can not'),
    ('couldnt', 'could not'),
    ('darent', 'dare not'),
    ('didnt', 'did not'),
    ('doesnt', 'does not'),
    ('dont', 'do not'),
    ('isnt', 'is not'),
    ('hasnt', 'has not'),
    ('havent', 'have not'),
    ('hadnt', 'had not'),
    ('maynt', 'may not'),
    ('mightnt', 'might not'),
    ('mustnt', 'must not'),
    ('neednt', 'need not'),
    ('oughtnt', 'ought not'),
    ('shant', 'shall not'),
    ('shouldnt', 'should not'),
    ('wasnt', 'was not'),
    ('werent', 'were not'),
    ('wont', 'will not'),
    ('wouldnt', 'would not'),
    ('aint', 'am not')
]



class DataExtractor(ABC):
    @abstractmethod
    def __init__(self, filename):
        pass

    @abstractmethod
    def process(self, outname):
        pass

class MichiganExtractor(DataExtractor): #Used for the Michigan tweet dataset, for the format [score\ttweet]
    def __init__(self, filename):
        self.filename = filename

    def process(self, outname):
        self.fileList = list(csv.reader(open(self.filename, 'r', encoding='utf-8'), delimiter='\t'))



class PreProcessor:
    def firstPass(self, sentence):
        sentence = sentence.lower().translate(str.maketrans('','',string.punctuation))
        for w in NEG_CONTRACTIONS:
            sentence = re.sub(w[0], w[1], sentence)
        return sentence.split()

    def stem(self, sentence):
        return

"""m = MichiganExtractor('../resources/raw/michigantraining.txt')
m.process('w/e')
p = PreProcessor()
for sentence in m.fileList:
    print(p.firstPass(sentence[1]))"""
