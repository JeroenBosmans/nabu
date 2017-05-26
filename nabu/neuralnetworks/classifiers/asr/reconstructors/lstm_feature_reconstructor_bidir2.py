'''@file lstm_reconstructor.py
contains the LstmReconstructor class'''

import tensorflow as tf
from nabu.neuralnetworks.classifiers.asr.reconstructors import reconstructor
from nabu.neuralnetworks.classifiers import layer

class LstmFeatureReconstructor(reconstructor.Reconstructor):
    '''
    A reconstructor that reconstructs the input features with an lstm chain.'''

    def reconstruct(self, hlfeat, reconstructor_inputs, is_training):
        '''

        This reconstructor approximates a number of input features with an LSTM
        RNN that has been given one hlf as initial state. It predicts the input
        features that were used to construct the next hlf.

        ADAPTED VERSION THAT MAKES IT POSSIBLE TO CHANGE THE NUMBER OF HIDDEN
        UNITS IN THE LSTM CELLS

        Args:
            hlfeat: the high level features that came out of the listener
                [batch_size x max_hl_seq_length x feat_dim]
            reconstructor_inputs: the feature inputs that are present as targets
                in a [batch_size x nbr_features x feature_dim] tensor
            is_training: boolean that keeps if we are currently training

        Returns:
            - an approximation of the input features
                [batch_size x number_audio_samples x num_quant_levels] tensor
        '''


        #get dimensions
        batch_size = int(hlfeat.get_shape()[0])
        max_nbr_features = int(hlfeat.get_shape()[1])
        max_nbr_input_features = int(reconstructor_inputs.get_shape()[1])
        input_dim = int(reconstructor_inputs.get_shape()[2])
        # use a trick to round up
        input_per_hl = -(-max_nbr_input_features//max_nbr_features)

        #create the rnn cell for the decoder
        rnn_cell = tf.contrib.rnn.BasicLSTMCell(self.number_units)

        # reform to the number of hidden units
        linear_layer1 = layer.Linear(self.number_units)
        hlfeat = linear_layer1(hlfeat, scope='layer1')

        # reshape high level features to do all computation in parallel
        hlfeat = tf.reshape(hlfeat, [-1, self.number_units])

        # add an all zero feature to the beginning, and remove the last feature
        zeros = tf.zeros([batch_size, 1, input_dim])
        reconstructor_inputs = tf.concat([zeros, reconstructor_inputs], 1)
        reconstructor_inputs = tf.slice(reconstructor_inputs, [0, 0, 0],
                                        [-1, max_nbr_input_features, -1])

        # pad with extra zero features if length is not equal to
        #input_per_hl*max_nbr_features
        if max_nbr_input_features < (max_nbr_features*input_per_hl):
            nbr_to_pad = (max_nbr_features*input_per_hl)-max_nbr_input_features
            zeros = tf.zeros([batch_size, nbr_to_pad, input_dim])
            reconstructor_inputs = tf.concat([reconstructor_inputs, zeros], 1)

        # reshape the reconstructor inputs to the correct format
        reconstructor_inputs = tf.reshape(reconstructor_inputs,
                                          [-1, input_per_hl, input_dim])

        # transform the inputs to a list
        inputs = tf.unstack(reconstructor_inputs, axis=1)

        #Make the initial state
        zero_hidden_state = rnn_cell.zero_state(
            batch_size*max_nbr_features, tf.float32).h
        init_state = tf.contrib.rnn.LSTMStateTuple(hlfeat, zero_hidden_state)

        output, _ = tf.contrib.legacy_seq2seq.rnn_decoder(
            decoder_inputs=inputs,
            initial_state=init_state,
            cell=rnn_cell,
            scope='rnn_decoder')

        #transform the output from a list to a tensor
        output_tensor = tf.stack(output, axis=1)

        # the dimension of the outputs of rnn may be different of the dimension
        # of the input features to approximate. Linear layer to convert this.
        linear_layer2 = layer.Linear(input_dim)
        logits = linear_layer2(output_tensor, scope='layer2')

        # transform back to a format of [batch_size x nbr_features x feat_dim]
        logits = tf.reshape(logits, [batch_size, -1, input_dim])

        # delete the samples that were simply padded before
        logits = tf.slice(logits, [0, 0, 0], [-1, max_nbr_input_features, -1])

        return logits
