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
        if 'samples_per_hlfeature' in conf:
            self.samples_per_hlfeature = int(conf['samples_per_hlfeature'])
        if 'unpredictable_samples' in conf:
            self.unpredictable_samples = int(conf['unpredictable_samples'])
        if 'reconstructor_numunits' in conf:
            self.number_units = int(conf['reconstructor_numunits'])
        self.output_dim = output_dim

        self.scope = tf.VariableScope(False, name or type(self).__name__)


    def __call__(self, hlfeat, reconstructor_inputs, is_training=False):
        '''
        Create the variables and do the forward computation

        Args:
            hlfeat: the high level features that came out of the listener
                [batch_size x max_hl_seq_length x feat_dim]
            reconstructor_inputs: the features that are used for reconstruction
                in a [batch_size x max_length x dim] tensor
            is_training: boolean that keeps if we are currently training

        Returns:
            - the output logits or approx of the reconstructed audio samples
                [batch_size x number_audio_samples x output_dim] tensor
        '''

        with tf.variable_scope(self.scope):

            reconstructed = self.reconstruct(hlfeat, reconstructor_inputs,
                                                is_training)

        self.scope.reuse_variables()

        return reconstructed

    @abstractmethod
    def reconstruct(self, hlfeat, reconstructor_inputs, is_training):
        '''
        Create the reconstructed features

        Args:
            hlfeat: the high level features that came out of the listener
                [batch_size x max_hl_seq_length x feat_dim]
            reconstructor_inputs: the features that are used for reconstruction
                in a [batch_size x max_length x dim] tensor
            is_training: boolean that keeps if we are currently training

        Returns:
            - the output logits or approx of the reconstructed audio samples
                [batch_size x number_audio_samples x output_dim] tensor
        '''
