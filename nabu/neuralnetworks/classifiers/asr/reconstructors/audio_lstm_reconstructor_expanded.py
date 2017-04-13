'''@file lstm_reconstructor.py
contains the LstmReconstructor class'''

import tensorflow as tf
import reconstructor

class AudioLstmReconstructor(reconstructor.Reconstructor):

    def reconstruct(self, hlfeat, reconstructor_inputs, is_training):
        '''
        Create the reconstructed audio samples

        This reconstructor uses one LSTM dynamic encoder to reconstruct the
        audio samples, when the inputs are formed by concatenation of the
        previous audio sample and the corresponding high level feature.

        Args:
            hlfeat: the high level features that came out of the listener
                [batch_size x max_hl_seq_length x feat_dim]
            reconstructor_inputs: the audio samples that are present as targets
                in a [batch_size x nbr_audiosamples x 1] tensor
            is_training: boolean that keeps if we are currently training

        Returns:
            - the output logits of the reconstructed audio samples
                [batch_size x number_audio_samples x num_quant_levels] tensor
        '''

        #get dimensions
        batch_size = int(hlfeat.get_shape()[0])
        max_nbr_features = int(hlfeat.get_shape()[1])
        hlf_dim = int(hlfeat.get_shape()[2])
        max_samples = int(reconstructor_inputs.get_shape()[1])

        #reshape the audio samples to the twodimenstional format and cast them
        # to integers
        reconstructor_inputs = tf.reshape(reconstructor_inputs,
                                            [batch_size,max_samples])
        reconstructor_inputs = tf.cast(reconstructor_inputs, tf.int32)

        #create the rnn cell for the decoder
        rnn_cell = tf.contrib.rnn.BasicLSTMCell(self.number_units)

        # add one zero high level feature vector to the beginning for each batch_size
        zeros = tf.zeros([batch_size, 1, hlf_dim])
        hlfeat = tf.concat([zeros, hlfeat],1)

        # the very last hlf for each batch can be discarded
        hlfeat = tf.slice(hlfeat,[0,0,0],[-1,max_nbr_features,-1])

        # repeat every hlf samples_per_hlfeature times to concat with inputs
        hlfeat = tf.expand_dims(hlfeat,0)
        hlfeat = tf.tile(hlfeat,[self.samples_per_hlfeature,1,1,1])
        hlfeat = tf.reshape(hlfeat, [batch_size, -1, hlf_dim])

        # remove the first samples, since the are unpredictable anyway
        reconstructor_inputs = tf.slice(reconstructor_inputs,
                                    [0,self.unpredictable_samples],[-1,-1])

        # add a zero in the beginning and remove the last one
        # this is done instead of working with sos and eos labels
        reconstructor_inputs = tf.slice(reconstructor_inputs,[0,0],
                                [-1,max_samples-self.unpredictable_samples-1])
        audio_samples = tf.concat([tf.zeros([batch_size,1], dtype=tf.int32),
                                                        reconstructor_inputs],1)

        #pad the inputs such that they are of length
        #max_nbr_features*samples_per_hlfeature
        nbr_audio_samples = int(audio_samples.get_shape()[1])
        nbr_needed = self.samples_per_hlfeature*max_nbr_features
        nbr_to_pad = nbr_needed-nbr_audio_samples
        zeros = tf.zeros([batch_size,nbr_to_pad], dtype=tf.int32)
        audio_samples = tf.concat([audio_samples,zeros],1)

        # concatenate the audiosamples and the high level feature to lstm inputs
        audio_samples = tf.expand_dims(audio_samples, 2)
        audio_samples = tf.cast(audio_samples, tf.float32)
        reconstructor_inputs = tf.concat([audio_samples, hlfeat],2)

        # as initial state we should just take the zeros state of the rnn cell
        zero_state = rnn_cell.zero_state(batch_size, tf.float32)

        dynamic_fn_train = tf.contrib.seq2seq.simple_decoder_fn_train(zero_state)
        sequence_lengths = tf.ones([batch_size],dtype=tf.int32)* \
                                    int(reconstructor_inputs.get_shape()[1])

        # compute the output by using a dynamic rnn_decoder
        # a static rnn decoder has way to many nodes for a fast graph creation
        output_tensor, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(
                                    decoder_fn=dynamic_fn_train,
                                    cell=rnn_cell,
                                    inputs=reconstructor_inputs,
                                    sequence_length=sequence_lengths)

        # we now have sequences of outputs of dimension of the high level features
        # still need to convert this to the dimension of the number of quantisation levels
        logits = tf.contrib.layers.linear(output_tensor, self.output_dim)

        #once again add the unpredictable samples, simply as all zeros
        zeros = tf.zeros([batch_size,self.unpredictable_samples,self.output_dim])
        logits=tf.concat([zeros, logits],axis=1)

        return logits
