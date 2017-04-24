'''@file normalizer_factory.py
Contains the normalizer factory
'''

from nabu.processing.target_normalizers import aurora4
from nabu.processing.target_normalizers import timit

def factory(normalizer_type):
    '''create a normalizer_type

    Args:
        normalizer_type: the type of normalizer_type

    Returns:
        a normalizer function'''

    if normalizer_type == 'aurora4_normalizer':
        return aurora4.Aurora4()
    elif normalizer_type == 'timit_phone_norm':
        return timit.Timit()
    else:
        raise Exception('Undefined normalizer: %s' % normalizer_type)
