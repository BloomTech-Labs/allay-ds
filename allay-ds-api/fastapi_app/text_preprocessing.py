"""Process incoming text into a format usable by the ML models.
"""

from fastapi_app import NLP


# Preprocessing for lemmatization
# add / remove stop words, normalize, any text processing as necessary

# additional tokens to ignore
STOP_WORDS = []

is_empty_pattern = re.compile(r'^\s*$')

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
                not token.is_stop
                and not token.is_punct
                and token.pos_ != 'PRON'
                and not is_empty_pattern.match(token.text)
                and len(token.lemma_) > 2
                and token.lemma_ not in STOP_WORDS
            ):
                doc_lemmas.append(token.lemma_)
        lemmas.append(doc_lemmas)
    return lemmas

