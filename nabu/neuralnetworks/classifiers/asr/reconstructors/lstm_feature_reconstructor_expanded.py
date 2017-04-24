'''@file lstm_reconstructor.py
contains the LstmReconstructor class'''

import tensorflow as tf
from nabu.neuralnetworks.classifiers.asr.reconstructors import reconstructor

class LstmFeatureReconstructor(reconstructor.Reconstructor):
    '''
    A reconstructor that reconstructs the input features with an lstm chain.'''

    def reconstruct(self, hlfeat, reconstructor_inputs, is_training):
        '''
        This reconstructor approximates the input features based on the high
        level features and the previous input features. This is done in one
        LSTM RNN decoder where the concatenation of input features with the
        previous corresponding are the inputs.

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
        hlf_dim = int(hlfeat.get_shape()[2])
        max_nbr_input_features = int(reconstructor_inputs.get_shape()[1])
        input_dim = int(reconstructor_inputs.get_shape()[2])
        # used a trick to round up
        input_per_hl = -(-max_nbr_input_features//max_nbr_features)

        #create the rnn cell for the decoder
        rnn_cell = tf.contrib.rnn.BasicLSTMCell(self.number_units)

        # add one zero hlf vector to the beginning for each batch_size
        zeros = tf.zeros([batch_size, 1, hlf_dim])
        hlfeat = tf.concat([zeros, hlfeat], 1)

        # the very last hlf for each batch can be discarded
        hlfeat = tf.slice(hlfeat, [0, 0, 0], [-1, max_nbr_features, -1])

        # repeat every hlf samples_per_hlfeature times to concat with inputs
        hlfeat = tf.expand_dims(hlfeat, 0)
        hlfeat = tf.tile(hlfeat, [input_per_hl, 1, 1, 1])
        hlfeat = tf.reshape(hlfeat, [batch_size, -1, hlf_dim])

        # add a zero in the beginning and remove the last one
        # this is done instead of working with sos and eos labels
        reconstructor_inputs = tf.slice(reconstructor_inputs, [0, 0, 0],
                                        [-1, max_nbr_input_features-1, -1])
        reconstructor_inputs = tf.concat([tf.zeros([batch_size, 1, input_dim]),
                                          reconstructor_inputs], 1)

        # pad with extra zero features if length is not equal to
        #input_per_hl*max_nbr_features
        if max_nbr_input_features < (max_nbr_features*input_per_hl):
            nbr_to_pad = (max_nbr_features*input_per_hl) \
                - max_nbr_input_features
            zeros = tf.zeros([batch_size, nbr_to_pad, input_dim])
            reconstructor_inputs = tf.concat([reconstructor_inputs, zeros], 1)

        # concatenate the hlf features with the original inputs
        reconstructor_inputs = tf.concat([hlfeat, reconstructor_inputs], 2)

        # as initial state we should just take the zeros state of the rnn cell
        zero_state = rnn_cell.zero_state(batch_size, tf.float32)

        dynamic_fn_train = tf.contrib.seq2seq.simple_decoder_fn_train(
            zero_state)
        sequence_lengths = tf.ones([batch_size], dtype=tf.int32)*\
            int(reconstructor_inputs.get_shape()[1])

        # compute the output by using a dynamic rnn_decoder
        # a static rnn decoder has way to many nodes for a fast graph creation
        output_tensor, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(
            decoder_fn=dynamic_fn_train,
            cell=rnn_cell,
            inputs=reconstructor_inputs,
            sequence_length=sequence_lengths)

        # we now have sequences of outputs of dimension of the hlfs
        #convert this to the dimension of the number of quantisation levels
        logits = tf.contrib.layers.linear(output_tensor, input_dim)

        # delete extra features that were just added by zero padding earlier
        if max_nbr_input_features < (max_nbr_features*input_per_hl):
            logits = tf.slice(logits, [0, 0, 0],
                              [-1, max_nbr_input_features, -1])

        return logits
