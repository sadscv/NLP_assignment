import sys
from gensim.models import word2vec
import logging
import gensim

def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    
    sentences = word2vec.Text8Corpus("wiki_seg.txt")
    model = word2vec.Word2Vec(sentences, size=200, alpha=0.025, window=5, min_count=5, workers=4)
    model.wv.save_word2vec_format('ch_wiki_200d.model.bin', binary=True)
    # how to load a model ?
    # model = word2vec.Word2Vec.load_word2vec_format("your_model.bin", binary=True)

def test(word):

    model = gensim.models.KeyedVectors.load_word2vec_format('ch_wiki_200d.model.bin', binary=True)
    result = model.most_similar(word)
    for e in result:
        print(e[0], e[1])
    



if __name__ == "__main__":
#    main()
    if len(sys.argv) != 2:
        print("Usage: python3 " + sys.argv[0] + " wiki_data_path")
        exit()
    word = sys.argv[1]
    test(word)
