"""Returns a rating using a CNN neural network.
"""

from random import uniform

from fastapi_app import MODEL


def get_score(text: str):
    """Returns a score of text content from the CNN.

    :param text: str, required.

    :return: float, Inappropriateness score - rating between 0.0 and 1.0,
    closer to 1.0 is more likely to be inappropriate.
    """
    return uniform(0.0, 1.0)
