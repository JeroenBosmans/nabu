'''@file reconstructor_factory
contains the reconstructor factory'''

from . import lstm_feature_reconstructor, audio_lstm_reconstructor, \
    audio_lstm_reconstructor_expanded, lstm_feature_reconstructor_expanded, \
    lstm_feature_reconstructor0, lstm_feature_reconstructor00

def factory(conf, output_dim):
    '''create a reconstructor

    Args:
        conf: the classifier config as a dictionary

    Returns:
        A reconstructor object'''

    if conf['reconstructor'] == 'lstm_feature_reconstructor':
        return lstm_feature_reconstructor.LstmFeatureReconstructor(
            conf, output_dim)
    elif conf['reconstructor'] == 'lstm_feature_reconstructor_bidir':
        return lstm_feature_reconstructor0.LstmFeatureReconstructor(
            conf, output_dim)
    elif conf['reconstructor'] == 'lstm_feature_reconstructor_bidir2':
        return lstm_feature_reconstructor00.LstmFeatureReconstructor(
            conf, output_dim)
    elif conf['reconstructor'] == 'lstm_feature_reconstructor_expanded':
        return lstm_feature_reconstructor_expanded.LstmFeatureReconstructor(
            conf, output_dim)
    elif conf['reconstructor'] == 'audio_lstm_reconstructor':
        return audio_lstm_reconstructor.AudioLstmReconstructor(
            conf, output_dim)
    elif conf['reconstructor'] == 'audio_lstm_reconstructor_expanded':
        return audio_lstm_reconstructor_expanded.AudioLstmReconstructor(
            conf, output_dim)
    else:
        raise Exception(
            'undefined asr reconstructor type: %s' % conf['reconstructor'])
