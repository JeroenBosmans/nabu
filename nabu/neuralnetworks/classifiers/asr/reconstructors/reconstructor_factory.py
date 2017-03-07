'''@file reconstructor_factory
contains the reconstructor factory'''

from . import lstm_reconstructor

def factory(conf, output_dim):
    '''create a reconstructor

    Args:
        conf: the classifier config as a dictionary

    Returns:
        A reconstructor object'''

    if conf['reconstructor'] == 'lstm_reconstructor':
        return lstm_reconstructor.LstmReconstructor(conf, output_dim)
    else:
        raise Exception('undefined asr reconstructor type: %s' % conf['reconstructor'])
