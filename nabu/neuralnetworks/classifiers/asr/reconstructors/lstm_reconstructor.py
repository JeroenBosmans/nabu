'''@file lstm_reconstructor.py
contains the LstmReconstructor class'''

import tensorflow as tf
import reconstructor
from nabu.neuralnetworks.classifiers import layer

class LstmReconstructor(reconstructor.Reconstructor):

    def reconstruct(self, hlfeat, reconstructor_inputs, is_training):
        '''
        Create the reconstructed audio samples

        Args:
            hlfeat: the high level features that came out of the listener
                [batch_size x max_hl_seq_length x feat_dim]
            sequence_length: a vector that contains the exact lenght of all
                of the high level feature vectors.
            is_training: boolean that keeps if we are currently training

        Returns:
            - the output logits of the reconstructed audio samples
                [batch_size x number_audio_samples x num_quant_levels] tensor
        '''
        #create the rnn cell for the decoder
        rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(int(hlfeat.get_shape()[2]))

        # transform reconstructor inputs to one hot encoded
        one_hot_inputs = tf.one_hot(reconstructor_inputs, self.output_dim,
                                    dtype=tf.float32)

        '''
        Hier loopen op een manier zodat we een voor een alle high level features
        verwerken en zo alle audio samples verwerken.
        '''

        # create the samples for the first PSEUDO high level feature (all zeros)
        pseudo_hlf = tf.zeros([hlfeat.get_shape()[0],hlfeat.get_shape()[2]])
        part_inputs = self.get_list_samples(one_hot_inputs, -1)
        last_output = tf.zeros([hlfeat.get_shape()[0],hlfeat.get_shape()[2]])
        output = self.reconstruct_one_part(rnn_cell, part_inputs, pseudo_hlf, last_output)

        print('dfasdfasdfasdf')
        print(hlfeat.get_shape()[1])
        # loop over all of the hl feature vectors
        #for i in range(hlfeat.get_shape()[1]):
        for i in range(5):
            part_inputs = self.get_list_samples(one_hot_inputs, i)
            hlf = tf.slice(hlfeat,[0,i,0],[-1,1,-1])
            hlf = tf.reshape(hlf, [int(hlfeat.get_shape()[0]),-1])
            temp = self.reconstruct_one_part(rnn_cell, part_inputs, hlf, last_output)
            tf.concat(1,[output, temp])
            print('in loop, iteration %d', i)

        # convert outputs from a list to a tensor
        output_tensor_time_major = tf.pack(output)

        #convert to the right axes
        output_tensor = tf.transpose(output_tensor_time_major, [1,0,2])

        # we now have sequence of outputs of dimension of the high level features
        # still need to convert this to the dimension of the number of quantisation levels
        linear_layer = layer.Linear(self.output_dim)
        logits = linear_layer(output_tensor)

        return logits



    def reconstruct_one_part(self, rnn_cell, decoder_inputs, hlf, last_output):
        '''
        Create a part of the audiosamples, all those which are computed from one
        high level feature

        Args:
            rnn_cel: the rnn cel that is the base of the decoder
            decoder_inputs: a list of inputs for the decoder, known from
                training data in a tensor [batch_size x num_quant_levels]
            hfl: the high level feature vector that we're using to predict
                this part of the audio samples.

        Returns:
            - the output logits for a part of the audio sampels to predict in
                a [batch_size x samples_per_hlfeature x num_quant_levels] tensor
        '''
        with tf.variable_scope(self.scope):
            output, _ = tf.nn.seq2seq.rnn_decoder(decoder_inputs=decoder_inputs,
                            initial_state=(hlf,last_output), cell=rnn_cell, scope='rnn_decoder')

        self.scope.reuse_variables()

        return output


    def get_list_samples(self, one_hot_inputs, hlf_number):
        '''
        Create a list of one hot inputs for a certain hlf that you're considering,
        represented by its number in the sequence (starting with zero)

        Args:
            one_hot_inputs: the tensor of all of the one hot inputs with shape
                            [batch_size x number_audio_samples x quant levels]
            hlf_number: the number of the hlf that is considered

        Returns:
            part_inputs_list: A list of samples_per_hlfeature one hot audio samples
        '''
        # compute the position of where to start for this hlf
        start = (hlf_number+1)*self.samples_per_hlfeature

        # slice the right part of the one hot inputs
        part_inputs = tf.slice(one_hot_inputs,[0,start,0],[-1,self.samples_per_hlfeature,-1])

        # transform to time major
        part_inputs_time_major = tf.transpose(part_inputs, [1, 0, 2])

        # transform to a list
        part_inputs_list = tf.unpack(part_inputs_time_major)

        return part_inputs_list
