'''@file reconstructor_factory
contains the reconstructor factory'''

from . import lstm_reconstructor, lstm_reconstructor_variant, lstm_reconstructor_variant2, lstm_reconstructor_dynamic, \
lstm_feature_reconstructor, audio_lstm_reconstructor, audio_lstm_reconstructor_expanded

def factory(conf, output_dim):
    '''create a reconstructor

    Args:
        conf: the classifier config as a dictionary

    Returns:
        A reconstructor object'''

    if conf['reconstructor'] == 'lstm_reconstructor':
        return lstm_reconstructor.LstmReconstructor(conf, output_dim)
    elif conf['reconstructor'] == 'lstm_reconstructor_variant':
        return lstm_reconstructor_variant.LstmReconstructorVariant(conf, output_dim)
    elif conf['reconstructor'] == 'lstm_reconstructor_variant2':
        return lstm_reconstructor_variant2.LstmReconstructorVariant2(conf, output_dim)
    elif conf['reconstructor'] == 'lstm_reconstructor_dynamic':
        return lstm_reconstructor_dynamic.LstmReconstructor(conf, output_dim)
    elif conf['reconstructor'] == 'lstm_feature_reconstructor':
        return lstm_feature_reconstructor.LstmFeatureReconstructor(conf, output_dim)
    elif conf['reconstructor'] == 'audio_lstm_reconstructor':
        return audio_lstm_reconstructor.AudioLstmReconstructor(conf, output_dim)
    elif conf['reconstructor'] == 'audio_lstm_reconstructor_expanded':
        return audio_lstm_reconstructor_expanded.AudioLstmReconstructor(conf, output_dim)
    else:
        raise Exception('undefined asr reconstructor type: %s' % conf['reconstructor'])
