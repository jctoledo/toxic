import gensim, logging
import os
import argparse


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()
def read_swear_words(afp):
    rm = set()
    fin = open(afp, 'r')
    for l in fin:
        rm.add(l.strip().lower())
    fin.close()
    return rm

def train_w2v_model(sentences):
    return gensim.models.Word2Vec(sentences, min_count=5)


def get_new_terms_from_w2v_model(model, swear_words):
    rm = set()
    for sw in swear_words:
        try:
            sim_tups = model.most_similar(sw)
            for st in sim_tups:
                rm.add(st[0])
        except KeyError as ke:
            logging.info(ke)
    return rm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--all_text", help="path to location of all text in corpus")
    parser.add_argument("-d", "--swear_dict", help="path to dictionary of swearing words")

    args = parser.parse_args()

    swear_words = read_swear_words(args.swear_dict)

    sentences = MySentences(args.all_text)
    # train word2vec on the two sentences
    model = train_w2v_model(sentences)
    added_swear_words = get_new_terms_from_w2v_model(model, swear_words)
    new_dict = added_swear_words.union(swear_words)
    fout = open("/tmp/swear_words_extended.txt", "w")
    for w in new_dict:
        fout.write(w+"\n")

    fout.close()
    w =2


