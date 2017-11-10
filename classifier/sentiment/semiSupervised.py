from sklearn.semi_supervised import LabelSpreading
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np

class SemiSupervised:
    def getD2VTrainVectors(self, combinendPath):
        self.model = Doc2Vec(dm=0, size=200, window=10, workers=4, iter=10)
        sentences = []
        with open(combinendPath, 'r') as combined:
            for line in combined:
                splitLine = line.split('\t')
                if len(splitLine) == 4:
                    sentences.append(TaggedDocument(words=splitLine[2].split(), tags=[splitLine[0]+' '+splitLine[1]]))
        print('finished getting sentences')
        self.model.build_vocab(sentences)
        self.model.train(sentences, total_examples=1386047, epochs=self.model.iter)
        print('trained doc2vec')
        dataset = []
        with open(combinendPath, 'r') as combined:
            counter = 0
            for line in combined:
                splitLine = line.rstrip('\n').split('\t')
                if len(splitLine) == 4:
                    dataset.append([self.model.docvecs[counter], splitLine[3]])
                    counter += 1

        print('got vectors from doc2vec')
        self.model.save('/tmp/twitter-model')
        self.model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
        return np.array(dataset)

if __name__ == '__main__':
    ss = SemiSupervised()
    data = ss.getD2VTrainVectors('../../resources/hydrated_tweets/Combined_TwitterData_v3.txt')
    print('got Vectors')
    model = LabelSpreading(kernel='rbf', max_iter=100, n_jobs=-1)
    model.fit(data)
    print('model fitted')
    vector = ss.model.infer_vector('fixing italys banks is helping europes economy heal'.split(), 'test1')
    print(model.predict(vector))