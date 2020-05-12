"""Process incoming text into a format usable by the ML models.
"""

from fastapi_app import NLP


# Preprocessing for lemmatization
# add / remove stop words, normalize, any text processing as necessary
# documents passed to the make_lemmas function should be processed with clean_strings first
#   df['cleaned'] = df['tweet'].apply(clean_strings)
#   df['lemmas'] = make_lemmas(nlp, df['cleaned']) 

import re

# https://github.com/dennybritz/cnn-text-classification-tf/blob/master/data_helpers.py
def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py

    modified to accept '@' and '#' characters
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`@#]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

# additional tokens to ignore
STOP_WORDS = ['user', 'amp', '-PRON-']

# empty / entirely whitespace
is_empty_pattern = re.compile(r'^\s*$')
# entirely (123, 1.23, 1/2, 1,234, 1st, 12th, etc)
is_numeric_pattern = re.compile(r'^[\d./,]+(th|st|am|pm)?$')
# entirely unicode symbols
is_symbol_pattern = re.compile(r'^[\d&#\\ud;]$')

def make_lemmas(nlp, docs):
    """Creates a list of documents containing the lemmas of each document in the input docs.

    :param nlp: spaCy NLP model to use
    :param docs: list of documents to lemmatize

    :returns: list of lemmatized documents
    """
    lemmas = []
    for doc in nlp.pipe(docs, batch_size=500):
        doc_lemmas = []
        for token in doc:
            if (
                not token.is_stop # spaCy stopwords
                and not token.is_punct # punctuation
                and token.pos_ != 'PRON' # pronouns
                and len(token.lemma_) > 2 # two or less characters
                and token.lemma_ not in STOP_WORDS # custom stopwords
                and not token.lemma_.startswith('@') # twitter handles
                and not token.lemma_.startswith('#') # hash tags
                and not is_empty_pattern.match(token.lemma_)
                and not is_numeric_pattern.match(token.lemma_)
                and not is_symbol_pattern.match(token.lemma_)
            ):
                doc_lemmas.append(token.lemma_)
        lemmas.append(doc_lemmas)
    return lemmas

