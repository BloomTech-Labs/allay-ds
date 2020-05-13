"""Returns a rating using a CNN neural network.
"""

from random import uniform

from numpy import reshape

from ..globals import MODEL
from ..text_preprocessing import preprocess_cnn

def get_score(text: str):
    """Returns a score of text content from the CNN.

    :param text: str, required.

    :return: float, Inappropriateness score - rating between 0.0 and 1.0,
    closer to 1.0 is more likely to be inappropriate.
    """
    processed = preprocess_cnn(text)
    y_pred = MODEL.predict(processed)
    return float(reshape(y_pred, (-1))[0])
