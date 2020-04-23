"""Returns a random rating regardless of content.

Placholder to return random ratings until model is ready to be deployed.
"""

from random import uniform


def get_score(text: str):
    """Returns a random score regardless of text content.

    :param text: str, required. Ignored.

    :return: float, Inappropriateness score - rating between 0.0 and 1.0,
    closer to 1.0 is more likely to be inappropriate.
    """
    return uniform(0.0, 1.0)
