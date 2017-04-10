'''@file encoder_reconstructor.py
contains a general encoder_reconstructor network'''

import tensorflow as tf
from nabu.neuralnetworks.classifiers import classifier
from encoders import encoder_factory
from reconstructors import reconstructor_factory

class EncoderReconstructor(classifier.Classifier):
    '''a general class for an encoder reconstructor system'''
    def __init__(self, conf, output_dim, name=None):
        '''LAS constructor

        Args:
            conf: The classifier configuration
            output_dim: the classifier output dimension
            name: the classifier name
        '''

        super(EncoderReconstructor, self).__init__(conf, output_dim, name)

        #create the listener
        self.encoder = encoder_factory.factory(conf)

        #create the reconstructor
        self.reconstructor = reconstructor_factory.factory(conf, self.output_dim)

    def _get_outputs(self, inputs, input_seq_length, targets=None,
                     target_seq_length=None, is_training=False):
        '''
        Add the neural net variables and operations to the graph

        Args:
            inputs: the inputs to the neural network, this is a
                [batch_size x max_input_length x feature_dim] tensor
            input_seq_length: The sequence lengths of the input utterances, this
                is a [batch_size] vector
            targets: the targets to the neural network, this is a tuple of
                text_targets [batch_size x max_target_length] tensor and
                reconstruction targets [batch_size x max_target_length2 x target_dim]
                tensor.
            target_seq_length: a tuple of [batch_size] tensors that represent the
                lenghts of the targets
            is_training: whether or not the network is in training mode

        Returns:
            A pair containing:
                - output logits (for the reconstruction only)
                - the output logits sequence lengths as a vector
        '''

        #add input noise
        std_input_noise = float(self.conf['std_input_noise'])
        if is_training and std_input_noise > 0:
            noisy_inputs = inputs + tf.random_normal(
                inputs.get_shape(), stddev=std_input_noise)
        else:
            noisy_inputs = inputs

        #compute the high level features
        hlfeat = self.encoder(
            inputs=noisy_inputs,
            sequence_lengths=input_seq_length,
            is_training=is_training)

        # take the second element of the targets tuple as your targets
        targets = targets[1]

        #compute the output logits
        logits = self.reconstructor(
            hlfeat=hlfeat,
            reconstructor_inputs=targets,
            is_training=is_training)

        return (None, logits), (None, target_seq_length[1])
