'''@file asr_decoder.py
contains the Reconstructor class'''

from abc import ABCMeta, abstractmethod
import tensorflow as tf

class Reconstructor(object):
    '''a general audio reconstructor object

    converts the high level features into a reconstruction of the audio signal'''

    __metaclass__ = ABCMeta

    def __init__(self, conf, output_dim, name=None):
        '''speller constructor

        Args:
            conf: the classifier config as a dictionary
            output_dim: the dimension of the expected output
            name: the reconstructor name'''


        #save the parameters
        self.conf = conf
        self.samples_per_hlfeature = int(conf['samples_per_hlfeature'])
        self.output_dim = output_dim

        self.scope = tf.VariableScope(False, name or type(self).__name__)


    def __call__(self, hlfeat, reconstructor_inputs, is_training=False):
        '''
        Create the variables and do the forward computation

        Args:
            hlfeat: the high level features that came out of the listener
                [batch_size x max_hl_seq_length x feat_dim]
            reconstructor_inputs: the audiosamples that we want to predict
                based high level features in [batch_size x audio_samples] tensor
            is_training: boolean that keeps if we are currently training

        Returns:
            - the output logits of the reconstructed audio samples
                [batch_size x number_audio_samples x output_dim] tensor
        '''

        with tf.variable_scope(self.scope):

            audiosamples = self.reconstruct(hlfeat, reconstructor_inputs,
                                                is_training)

        self.scope.reuse_variables()

        return audiosamples

    @abstractmethod
    def reconstruct(self, hlfeat, reconstructor_inputs, is_training):
        '''
        Als alles werkt deze comments kopieren van hierboven?


        Create the reconstructed audio samples

        Args:
            hlfeat: the high level features that came out of the listener
                [batch_size x max_hl_seq_length x feat_dim]
            sequence_length: a vector that contains the exact lenght of all
                of the high level feature vectors.
            num_quant_levels: the number of quantisation levels in the
                audiosamples that will be reconstructed
            samples_per_hlfeature: the number of audiosamples that will be
                reconstructed with one high level feature
            is_training: boolean that keeps if we are currently training

        Returns:
            - the output logits of the reconstructed audio samples
                [batch_size x number_audio_samples x num_quant_levels] tensor
        '''