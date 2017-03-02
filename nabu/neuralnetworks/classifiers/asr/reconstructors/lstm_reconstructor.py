'''@file lstm_reconstructor.py
contains the LstmReconstructor class'''

import tensorflow as tf
import reconstructor

class LstmReconstructor(reconstructor.Reconstructor):

    def reconstruct(self, hlfeat, sequence_lengths, is_training):
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

        '''
        We moeten hier ook toegang hebben tot het originele audio signaal!!!
        '''

        #create the rnn cell for the decoder
        rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(self.num_quant_levels)
        '''
        Hier loopen op een manier zodat we een voor een alle high level features
        verwerken en zo alle audio samples verwerken.
        '''
        # create variable for output tensor
#        output = tf.zeros([hlfeat.get_shape[0],
#                            self.samples_per_hlfeature*hlfeat.get_shape[1],
#                                self.num_quant_levels])
        decoder_inputs = 0

        output = self.reconstruct_one_part(rnn_cell, decoder_inputs, tf.slice(rnn_cell, decoder_inputs, hlfeat,[0,0,0],[-1,1,-1]))

        # loop over all of the hl feature vectors
        for i in range(1,hlfeat.get_shape[1]):
            temp = self.reconstruct_one_part(rnn_cell, decoder_inputs, tf.slice(hlfeat,[0,i,0],[-1,1,-1]))
            tf.concat([output, temp], 1)



    def reconstruct_one_part(self, rnn_cell, decoder_inputs, hlf):
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
        output, _ = tf.nn.seq2seq.rnn_decoder(decoder_inputs=decoder_inputs,
                initial_state=hlf, cell=rnn_cell, scope='rnn_decoder')

        return output
