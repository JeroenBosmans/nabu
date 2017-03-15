'''@file dblstm.py
contains de LAS class'''

import tensorflow as tf
from nabu.neuralnetworks.classifiers import classifier
from encoders import encoder_factory
from asr_decoders import asr_decoder_factory
from reconstructors import reconstructor_factory

class EncoderDecoderReconstructor(classifier.Classifier):
    '''a general class for an encoder decoder reconstructor system'''
    def __init__(self, conf, output_dim, name=None):
        '''Constructor for this kind of object

        Args:
            conf: The classifier configuration
            output_dim: the classifier output dimension as a tuple of the dimension
                of the actual outputs and the dimension of the quantized reconstructed audio samples
            name: the classifier name
        '''

        super(EncoderDecoderReconstructor, self).__init__(conf, output_dim, name)

        #create the listener
        self.encoder = encoder_factory.factory(conf)

        #create the speller
        self.decoder = asr_decoder_factory.factory(conf, self.output_dim[0])

        #create the reconstructors
        self.reconstructor = reconstructor_factory.factory(conf, self.output_dim[1])

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
                [batch_size x max_output_length] tensors. The targets can be
                used during training. The first element are the text targets,
                the second element are the quantized audio samples
            target_seq_length: The sequence lengths of the target utterances,
                this is a tuple [batch_size] vectors. Firste element is for the
                text targets, second is for the quantized audio samples
            is_training: whether or not the network is in training mode

        Returns:
            A pair containing:
                - output logits as a tuple of text targets and reconstructed audio samples
                - the output logits sequence lengths as a tuple of two vectors
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

        #prepend a sequence border label to the targets to get the encoder
        #inputs, the label is the last label
        batch_size = int(targets[0].get_shape()[0])
        s_labels = tf.constant(self.output_dim[0]-1,
                               dtype=tf.int32,
                               shape=[batch_size, 1])
        encoder_inputs = tf.concat([s_labels, targets[0]],1)

        #compute the output logits
        text_logits, _ = self.decoder(
            hlfeat=hlfeat,
            encoder_inputs=encoder_inputs,
            initial_state=self.decoder.zero_state(batch_size),
            first_step=True,
            is_training=is_training)

        #for the reconstruction, we need the audio samples as targets
        #add sos labels to the target sequence
        batch_size = int(targets[1].get_shape()[0])
        sos_labels = tf.constant(self.output_dim[1]-1,
                               dtype=tf.int32,
                               shape=[batch_size, 1])
        reconstructor_inputs = tf.concat([sos_labels, targets[1]],1)

        #compute the output logits
        audio_logits = self.reconstructor(
            hlfeat=hlfeat,
            reconstructor_inputs=reconstructor_inputs,
            is_training=is_training)

        #assemble two kind of logits and lengths in tuples
        logits = (text_logits, audio_logits)
        logits_lengths = (target_seq_length[0]+1,target_seq_length[1]+1)

        return logits, logits_lengths
