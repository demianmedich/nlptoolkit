# coding=utf-8
from typing import Union


def convert_to_unicode(text: Union[str, bytes]):
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))
