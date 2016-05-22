from gensim.models.word2vec import Word2Vec


def load_word2vec(w2v):
    if isinstance(w2v, str):
        print("Loading word vectors from '%s'..." % w2v, flush=True)
        try:
            w2v = Word2Vec.load_word2vec_format(w2v)
        except ValueError:
            w2v = Word2Vec.load_word2vec_format(w2v, binary=True)
    return w2v
