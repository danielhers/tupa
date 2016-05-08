from gensim.models.word2vec import Word2Vec


def load_word2vec(file):
    if isinstance(file, str):
        print("Loading word vectors from '%s'..." % file, flush=True)
        return Word2Vec.load_word2vec_format(file)
    return file
