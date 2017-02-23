'''@file listener.py
contains the listener code'''

import tensorflow as tf
import encoder
from nabu.neuralnetworks.classifiers import layer

class Unidir_Listener(encoder.Encoder):
    '''a listener object that uses unidirectional LSTM's in stead of BLSTMs.

    transforms input features into a high level representation'''

    def __init__(self, conf, name=None):
        '''Listener constructor

        Args:
            numlayers: the number of PBLSTM layers
            numunits: the number of units in each layer
            dropout: the dropout rate
            name: the name of the Listener'''

        #create the plstm layer
        self.plstm = layer.PLSTMLayer(int(conf['listener_numunits']))

        #create the lstm layer
        self.lstm = layer.LSTMLayer(int(conf['listener_numunits']))

        super(Unidir_Listener, self).__init__(conf, name)

    def encode(self, inputs, sequence_lengths, is_training=False):
        '''
        Create the variables and do the forward computation

        Args:
            inputs: the input to the layer as a
                [batch_size, max_length, dim] tensor
            sequence_length: the length of the input sequences
            is_training: whether or not the network is in training mode

        Returns:
            the output of the layer as a [batch_size, max_length, output_dim]
            tensor
        '''

        outputs = inputs
        output_seq_lengths = sequence_lengths
        for l in range(int(self.conf['listener_numlayers'])):
            outputs, output_seq_lengths = self.plstm(
                outputs, output_seq_lengths, 'layer%d' % l)

            if float(self.conf['listener_dropout']) < 1 and is_training:
                outputs = tf.nn.dropout(
                    outputs, float(self.conf['listener_dropout']))

        outputs = self.lstm(
            outputs, output_seq_lengths,
            'layer%d' % int(self.conf['listener_numlayers']))

        if float(self.conf['listener_dropout']) < 1 and is_training:
            outputs = tf.nn.dropout(outputs,
                                    float(self.conf['listener_dropout']))

        return outputs
