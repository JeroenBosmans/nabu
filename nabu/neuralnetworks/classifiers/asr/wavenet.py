'''@file wavenet.py
a wavenet classifier'''

import tensorflow as tf
from nabu.neuralnetworks.classifiers import classifier, layer

class Wavenet(classifier.Classifier):
    ''''a wavenet classifier'''

    def _get_outputs(self, inputs, input_seq_length, targets=None,
                     target_seq_length=None, is_training=False):
        '''
        Add the neural net variables and operations to the graph

        Args:
            inputs: the inputs to the neural network, this is a
                [batch_size x max_input_length x feature_dim] tensor
            input_seq_length: The sequence lengths of the input utterances, this
                is a [batch_size] vector
            targets: the targets to the neural network, this is a
                [batch_size x max_output_length] tensor. The targets can be
                used during training
            target_seq_length: The sequence lengths of the target utterances,
                this is a [batch_size] vector
            is_training: whether or not the network is in training mode

        Returns:
            A pair containing:
                - output logits
                - the output logits sequence lengths as a vector
        '''

        #create the gated convolutional layers
        dconv = layer.GatedAConv1d(int(self.conf['kernel_size']))

        #create the one by one convolution layer
        onebyone = layer.Conv1dLayer(int(self.conf['num_units']), 1, 1)

        #create the output layer
        outlayer = layer.Conv1dLayer(int(self.conf['num_units']), 1, 1)

        #add gaussian noise to the inputs if training
        if is_training:
            forward = inputs + tf.random_normal(inputs.get_shape(),
                                                stddev=0.6)
        else:
            forward = inputs

        #apply the input layer
        logits = 0
        forward = onebyone(forward, input_seq_length, is_training,
                           'inlayer')
        forward = tf.nn.tanh(forward)

        #apply the the blocks of dilated convolutions layers
        for b in range(int(self.conf['num_blocks'])):
            for l in range(int(self.conf['num_layers'])):
                forward, highway = dconv(
                    forward, input_seq_length, self.conf['causal'] == 'True'
                    , 2**l, is_training, 'dconv%d-%d' % (b, l))
                logits += highway

        #apply the relu
        logits = tf.nn.relu(logits)

        #apply the one by one convloution
        logits = onebyone(logits, input_seq_length, is_training, '1x1')
        logits = tf.nn.relu(logits)

        #apply the output layer
        logits = outlayer(logits, input_seq_length, is_training, 'outlayer')


        return logits, input_seq_length
