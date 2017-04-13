'''@file lstm_reconstructor.py
contains the LstmReconstructor class'''

import tensorflow as tf
import reconstructor
from nabu.neuralnetworks.classifiers import layer

class AudioLstmReconstructor(reconstructor.Reconstructor):

    def reconstruct(self, hlfeat, reconstructor_inputs, is_training):
        '''
        Create the reconstructed audio samples

        This reconstructor uses a LSTM RNN to transform each of the high level
        features (given as initial state to this RNN) to a sequence of the number
        of audiosamples that were used to construct the next high level feature.

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
        rnn_cell = tf.contrib.rnn.BasicLSTMCell(int(hlfeat.get_shape()[2]))

        # add one zero hlf vector to the beginning for each batch_size
        zeros = tf.zeros([batch_size, 1, hlf_dim])
        hlfeat = tf.concat([zeros, hlfeat],1)

        # the very last hlf for each batch can be discarded
        hlfeat = tf.slice(hlfeat,[0,0,0],[-1,max_nbr_features,-1])

        # reshape high level features to do all computation in parallel
        hlfeat = tf.reshape(hlfeat,[-1, hlf_dim])

        # remove the first samples, since the are unpredictable anyway
        reconstructor_inputs = tf.slice(reconstructor_inputs,
                                        [0,self.unpredictable_samples],[-1,-1])

        # add a zero in the beginning and remove the last one
        # this is done instead of working with sos and eos labels
        reconstructor_inputs = tf.slice(reconstructor_inputs,[0,0],
                                [-1,max_samples-self.unpredictable_samples-1])
        reconstructor_inputs = tf.concat(
            [tf.zeros([batch_size,1], dtype=tf.int32), reconstructor_inputs],1)

        #pad the inputs such that they are of length
        #max_nbr_features*samples_per_hlfeature
        nbr_audio_samples = int(reconstructor_inputs.get_shape()[1])
        nbr_needed = self.samples_per_hlfeature*max_nbr_features
        nbr_to_pad = nbr_needed-nbr_audio_samples
        zeros = tf.zeros([batch_size,nbr_to_pad], dtype=tf.int32)
        reconstructor_inputs = tf.concat([reconstructor_inputs,zeros],1)

        # reshape the reconstructor inputs to the correct format
        reconstructor_inputs = tf.reshape(reconstructor_inputs,
                                            [-1, self.samples_per_hlfeature])

        # transform reconstructor inputs to one hot encoded
        one_hot_inputs = tf.one_hot(reconstructor_inputs, self.output_dim,
                                    dtype=tf.float32)

        # transform the inputs to a list
        inputs = tf.unstack(one_hot_inputs, axis=1)

        #as initial previous inputs, for now we take zero vectors
        last_output = tf.zeros([hlfeat.get_shape()[0],hlfeat.get_shape()[1]])

        output, _ = tf.contrib.legacy_seq2seq.rnn_decoder(
                                        decoder_inputs=inputs,
                                        initial_state=(hlfeat,last_output),
                                        cell=rnn_cell,
                                        scope='rnn_decoder')

        #transform the output from a list to a tensor
        output_tensor = tf.stack(output, axis=1)

        # we now have sequences of outputs of dimension of the hlfs
        # convert this to the dimension of the number of quantization levels
        linear_layer = layer.Linear(self.output_dim)
        logits = linear_layer(output_tensor)

        # transform back to a format of
        #[batch_size x nbr_audio_samples x quantlevels]
        logits = tf.reshape(logits,[batch_size,-1,self.output_dim])

        #once again add the unpredictable samples, simply as all zeros
        zeros = tf.zeros([batch_size,self.unpredictable_samples,self.output_dim])
        logits=tf.concat([zeros, logits],axis=1)

        return logits
