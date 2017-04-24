'''@file feature_computer_factory.py
contains the FeatureComputer factory'''

from nabu.processing.feature_computers import mfcc
from nabu.processing.feature_computers import fbank
from nabu.processing.feature_computers import quant_audio

def factory(conf):
    '''
    create a FeatureComputer

    Args:
        conf: the feature configuration
    '''

    if conf['feature'] == 'fbank':
        return fbank.Fbank(conf)
    elif conf['feature'] == 'mfcc':
        return mfcc.Mfcc(conf)
    elif conf['feature'] == 'quant_audio':
        return quant_audio.Quant_Audio(conf)
    else:
        raise Exception('Undefined feature type: %s' % conf['feature'])
