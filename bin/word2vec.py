import gensim, logging
import os

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()

if __name__ == '__main__':

    sentences = MySentences("/home/jose/Documents/Data/Toxic/sample")
    # train word2vec on the two sentences
    model = gensim.models.Word2Vec(sentences, min_count=5)
    w =2


