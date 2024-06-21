from sklearn.feature_extraction.text import TfidfVectorizer

# Vectorization parameters
NGRAM_RANGE = (1, 1)
TOP_K = 1678
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
    # Map each token to a unique integer in the range [1, 1678]
    token_to_int = {token: idx + 1 for idx, token in enumerate(tokens)}

    return dataOut, tokens, vectorizer, token_to_int