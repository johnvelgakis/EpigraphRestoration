from sklearn.feature_extraction.text import TfidfVectorizer

# Vectorization parameters
NGRAM_RANGE = (1, 2)
TOP_K = 2000
TOKEN_MODE = 'word'

# Vectorize texts using TF-IDF
def vectorize_texts(train_texts):
    kwargs = {
        'ngram_range': NGRAM_RANGE,
        'dtype': 'float32',
        'strip_accents': 'unicode',
        'decode_error': 'replace',
        'analyzer': TOKEN_MODE,
        'max_df': 0.9,
        'norm': 'l2',
        'max_features': TOP_K,
    }
    vectorizer = TfidfVectorizer(**kwargs)
    dataOut = vectorizer.fit_transform(train_texts)
    tokens = vectorizer.get_feature_names_out()
    return dataOut, tokens, vectorizer