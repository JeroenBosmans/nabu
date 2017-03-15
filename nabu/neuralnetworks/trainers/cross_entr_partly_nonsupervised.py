'''@file cross_enthropytrainer_rec.py
contains the CrossEnthropyTrainerRec for reconstruction of the audio samples'''

import tensorflow as tf
import trainer
from nabu.neuralnetworks import ops

class CrossEntropyTrainerNonsupervised(trainer.Trainer):
    '''A trainer that minimises the cross-enthropy loss, the output sequences
    must be of the same length as the input sequences'''

    def compute_loss(self, targets, logits, logit_seq_length,
                     target_seq_length):
        '''
        Compute the loss

        Creates the operation to compute the cross-entropy loss for every input
        frame (if you want to have a different loss function, overwrite this
        method)

        Args:
            targets: a tupple of targets, the first one being a
                [batch_size, max_target_length] tensor containing the real
                targets, the second one being a [batch_size, max_audioseq_length]
                tensor containing the audio samples or other extra information.
            logits: a tuple of  [batch_size, max_logit_length, dim] tensors
                containing the logits for the text and the audio samples
            logit_seq_length: the length of all the logit sequences as a tuple of
                [batch_size] vectors
            target_seq_length: the length of all the target sequences as a
                tupple of two [batch_size] vectors, both for one of the elements
                in the targets tupple

        Returns:
            a scalar value containing the loss
        '''

        with tf.name_scope('cross_entropy_loss'):
            output_dim_text = int(logits[0].get_shape()[2])
            output_dim_audio = int(logits[1].get_shape()[2])

            # get the tradeoff parameters
            tradeoff = float(self.conf['loss_trade_off'])
            if tradeoff<0 or tradeoff>1:
                raise Exception('Trade off parameter for the loss function should be between 0 and 1')

            #put all the text targets on top of each other
            split_text_targets = tf.unstack(targets[0])
            for i, target in enumerate(split_text_targets):
                #only use the real data
                split_text_targets[i] = target[:target_seq_length[0][i]]

                #append an end of sequence label
                split_text_targets[i] = tf.concat(
                    [split_text_targets[i], [output_dim_text-1]],0)

            #put all the audio targets on top of each other
            split_audio_targets = tf.unstack(targets[0])
            for i, target in enumerate(split_audio_targets):
                #only use the real data
                split_audio_targets[i] = target[:target_seq_length[0][i]]

                #append an end of sequence label
                split_audio_targets[i] = tf.concat(
                    [split_audio_targets[i], [output_dim_audio-1]],0)

            #concatenate the text targets
            nonseq_text_targets = tf.concat(split_text_targets,0)

            #concatenate the audio targets
            nonseq_audio_targets = tf.concat(split_audio_targets,0)

            #convert the text logits to non sequential data
            nonseq_text_logits = ops.seq2nonseq(logits[0], logit_seq_length[0])

            #convert the audio logits to non sequential data
            nonseq_audio_logits = ops.seq2nonseq(logits[1], logit_seq_length[1])

            #one hot encode the text targets
            #pylint: disable=E1101
            nonseq_text_targets = tf.one_hot(nonseq_text_targets, output_dim_text)

            #one hot encode the audio targets
            #pylint: disable=E1101
            nonseq_audio_targets = tf.one_hot(nonseq_audio_targets, output_dim_audio)

            #compute the cross-enthropy loss for the prediction
            loss_prediction = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=nonseq_text_logits, labels=nonseq_text_targets))

            #compute the cross-enthropy loss for the recosntruction
            loss_reconstruction = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=nonseq_audio_logits, labels=nonseq_audio_targets))

            # make a combination of the two loss functions
            loss = tradeoff*loss_prediction + (1-tradeoff)*loss_reconstruction

        return loss
